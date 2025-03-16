import Honeycrisp
import Testing

@testable import CMA

@Test func testRosenbrock() async throws {
  func rosenbrock(_ coords: Tensor, a: Float = 1.0, b: Float = 100.0) -> Tensor {
    let x = coords[..., 0]
    let y = coords[..., 1]
    return (a - x).pow(2) + b * (y - x.pow(2)).pow(2)
  }
  let solver = CMA(config: .init(), initialValue: Tensor(data: [0.0, 0.0]))
  for _ in 0..<1000 {
    let populationSample = solver.sample()
    let evals = rosenbrock(populationSample)
    solver.update(samples: populationSample, scores: evals)
  }
  let minimum = solver.mean
  let minVec = try await minimum.floats()
  let actualMinimum = Tensor(data: [1.0, 1.0])
  let delta = try await (minimum - actualMinimum).abs().max().item()
  #expect(delta < 0.0001, "expected minimum (1.0, 1.0) but got \(minVec)")
}
