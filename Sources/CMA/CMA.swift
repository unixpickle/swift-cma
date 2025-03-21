import Foundation
import HCBacktrace
import Honeycrisp

public class CMA {
  public struct Config: Sendable, Codable {
    public let stepSize: Float
    public let population: Int?
    public let recombinationFrac: Float
    public let samplesPerEig: Int?

    public init(
      stepSize: Float = 0.5, population: Int? = nil, recombinationFrac: Float = 0.5,
      samplesPerEig: Int? = nil
    ) {
      self.stepSize = stepSize
      self.population = population
      self.recombinationFrac = recombinationFrac
      self.samplesPerEig = samplesPerEig
    }
  }

  public struct State: Codable {
    public let mean: TensorState
    public let sigma: TensorState
    public let pathC: TensorState
    public let pathSigma: TensorState
    public let basis: TensorState
    public let eigVals: TensorState
    public let covariance: TensorState
    public let covarianceInvSqrt: TensorState
    public let evalCount: Int
    public let evalCountAtLastEig: Int
  }

  private let dimCount: Int
  private let config: Config
  private let population: Int
  private let recombinationCount: Int
  private let weights: Tensor  // [recombinationCount]

  private let varianceEffectiveness: Float
  private let timeConstantC: Float
  private let timeConstantSigma: Float
  private let lrRank1: Float
  private let lrRankRecombination: Float
  private let sigmaDamping: Float
  private let expectedNorm: Float
  private let samplesPerEig: Int

  public var mean: Tensor  // [N]
  public var sigma: Tensor  // []
  private var pathC: Tensor  // [N]
  private var pathSigma: Tensor  // [N]
  private var basis: Tensor  // [N x N]
  private var eigVals: Tensor  // [N]
  private var covariance: Tensor  // [N x N]
  private var covarianceInvSqrt: Tensor  // [N x N]

  private var evalCount: Int = 0
  private var evalCountAtLastEig: Int = 0

  public var covarianceCondition: Tensor {
    let lowest = eigVals.min()
    let highest = eigVals.max()
    return (highest / lowest).pow(2)
  }

  public init(config: Config, initialValue: Tensor) {
    mean = initialValue
    sigma = Tensor(data: [config.stepSize], shape: [])

    dimCount = initialValue.shape[0]
    let N = Float(dimCount)
    self.config = config
    population = config.population ?? 4 + Int(floor(3 * log(Double(dimCount))))
    recombinationCount = max(
      1, Int(round(Double(config.recombinationFrac) * Double(population))))
    let bias = log(Float(recombinationCount) + 0.5)
    var w = (1...recombinationCount).map { i in bias - log(Float(i)) }
    let wSum = w.reduce(0.0, +)
    w = w.map { $0 / wSum }
    weights = Tensor(data: w)

    varianceEffectiveness = pow(w.reduce(0.0, +), 2) / w.map { pow($0, 2) }.reduce(0.0, +)
    timeConstantC =
      (4 + varianceEffectiveness / N)
      / (N + 4 + 2 * varianceEffectiveness / N)
    timeConstantSigma =
      (varianceEffectiveness + 2) / (N + varianceEffectiveness + 5)
    lrRank1 = 2.0 / (pow(N + 1.3, 2) + varianceEffectiveness)
    lrRankRecombination = min(
      1 - lrRank1,
      2 * (varianceEffectiveness - 2.0 + 1.0 / varianceEffectiveness)
        / (pow(N + 2.0, 2) + varianceEffectiveness)
    )
    sigmaDamping =
      1 + 2 * max(0.0, sqrt((varianceEffectiveness - 1) / (N + 1)) - 1)
      + timeConstantSigma

    expectedNorm = sqrt(N) * (1 - 1.0 / (4.0 * N) + 1.0 / (21.0 * pow(N, 2)))

    samplesPerEig =
      config.samplesPerEig ?? Int(Float(population) / (lrRank1 + lrRankRecombination) / N / 10.0)

    pathC = Tensor(zeros: [dimCount])
    pathSigma = pathC
    basis = Tensor(identity: dimCount)
    eigVals = Tensor(ones: [dimCount])
    covariance = basis &* Tensor.diagonal(eigVals.pow(2)) &* basis.t()
    covarianceInvSqrt = basis &* Tensor.diagonal(1 / eigVals) &* basis.t()
  }

  @recordCaller private func _sample() -> Tensor {
    let noise = Tensor(randn: [population, dimCount])
    return mean + sigma * ((eigVals * noise) &* basis.t())
  }

  @recordCaller private func _update(samples: Tensor, scores: Tensor) {
    #alwaysAssert(
      samples.shape == [population, dimCount], "invalid samples shape: \(samples.shape)")
    #alwaysAssert(
      scores.shape.count == 1 && scores.shape[0] == samples.shape[0],
      "invalid scores shape: \(scores.shape)")

    evalCount += samples.shape[0]

    let N = Float(dimCount)
    let indices = scores.argsort(axis: 0)[..<recombinationCount]
    let chosenSamples = samples.gather(axis: 0, indices: indices)

    let oldMean = mean
    mean = (chosenSamples * weights[..., NewAxis()]).sum(axis: 0)

    let pathSigmaTerm1 = (1 - timeConstantSigma) * pathSigma
    let pathSigmaTerm2 = sqrt(
      timeConstantSigma * (Float(2.0) - timeConstantSigma) * varianceEffectiveness)
    let pathSigmaTerm3 = Backtrace.record {
      (covarianceInvSqrt &* (mean - oldMean)[..., NewAxis()] / sigma).squeeze(
        axis: 1)
    }
    pathSigma = pathSigmaTerm1 + pathSigmaTerm2 * pathSigmaTerm3
    let pathSigmaNorm = pathSigma.pow(2).sum().sqrt()

    let threshold = 1.4 + 2 / (N + 1)
    let hSigDelta = pow(1 - timeConstantSigma, 2 * Float(evalCount) / Float(population))
    let hSig: Tensor = ((pathSigmaNorm / sqrt(Float(1.0) - hSigDelta)) / expectedNorm < threshold)
      .cast(.float32)
    pathC =
      (1 - timeConstantC) * pathC + hSig * (pathC * (2 - pathC) * varianceEffectiveness).sqrt()
      * (mean - oldMean) / sigma

    let artmp = (chosenSamples - oldMean) / sigma
    let cTerm1 = (1 - lrRank1 - lrRankRecombination) * covariance
    let cTerm2 =
      lrRank1
      * (Tensor.outer(pathC, pathC) + (1 - hSig) * timeConstantC * (2 - timeConstantC) * covariance)
    let cTerm3 = Backtrace.record {
      lrRankRecombination * (artmp.t() &* Tensor.diagonal(weights) &* artmp)
    }
    covariance = cTerm1 + cTerm2 + cTerm3

    sigma = sigma * ((timeConstantSigma / sigmaDamping) * (pathSigmaNorm / expectedNorm - 1)).exp()

    if evalCount - evalCountAtLastEig > samplesPerEig {
      evalCountAtLastEig = evalCount
      let (u, s, _) = covariance.svd()

      // This should do nothing, but it technically enforces symmetry
      covariance = u &* Tensor.diagonal(s) &* u.t()

      eigVals = s.sqrt()
      basis = u
      covarianceInvSqrt = basis &* Tensor.diagonal(1 / eigVals) &* basis.t()
    }
  }

  public var state: TracedBlock<State> {
    let mean = mean
    let sigma = sigma
    let pathC = pathC
    let pathSigma = pathSigma
    let basis = basis
    let eigVals = eigVals
    let covariance = covariance
    let covarianceInvSqrt = covarianceInvSqrt
    let evalCount = evalCount
    let evalCountAtLastEig = evalCountAtLastEig
    return TracedBlock {
      State(
        mean: try await mean.state(),
        sigma: try await sigma.state(),
        pathC: try await pathC.state(),
        pathSigma: try await pathSigma.state(),
        basis: try await basis.state(),
        eigVals: try await eigVals.state(),
        covariance: try await covariance.state(),
        covarianceInvSqrt: try await covarianceInvSqrt.state(),
        evalCount: evalCount,
        evalCountAtLastEig: evalCountAtLastEig
      )
    }
  }

  @recordCaller private func _loadState(_ state: State) throws {
    mean = Tensor(state: state.mean)
    sigma = Tensor(state: state.sigma)
    pathC = Tensor(state: state.pathC)
    pathSigma = Tensor(state: state.pathSigma)
    basis = Tensor(state: state.basis)
    eigVals = Tensor(state: state.eigVals)
    covariance = Tensor(state: state.covariance)
    covarianceInvSqrt = Tensor(state: state.covarianceInvSqrt)
    evalCount = state.evalCount
    evalCountAtLastEig = state.evalCountAtLastEig
  }
}
