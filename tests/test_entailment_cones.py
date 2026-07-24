import math
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EntailmentConeTests(unittest.TestCase):
    def test_apex_angle_distinguishes_outward_and_inward_same_ray(self):
        from src.system1.entailment_cones import apex_angle, origin_angle

        apex = torch.tensor([[0.5, 0.0]])
        outward = torch.tensor([[0.7, 0.0]])
        inward = torch.tensor([[0.3, 0.0]])
        self.assertAlmostEqual(apex_angle(apex, outward).item(), 0.0, places=6)
        self.assertAlmostEqual(
            apex_angle(apex, inward).item(), math.pi, places=6
        )
        self.assertAlmostEqual(origin_angle(apex, outward).item(), 0.0, places=6)
        self.assertAlmostEqual(origin_angle(apex, inward).item(), 0.0, places=6)

    def test_inverse_cone_accepts_premise_to_specific_theorem(self):
        from src.system1.entailment_cones import (
            cone_energy,
            cone_rank_scores,
            inverse_cone_energy,
        )

        premise = torch.tensor([[0.3, 0.0]])
        theorem = torch.tensor([[0.7, 0.0]])
        self.assertAlmostEqual(
            inverse_cone_energy(theorem, premise).item(), 0.0, places=6
        )
        self.assertGreater(cone_energy(theorem, premise).item(), 3.0)
        scores, membership = cone_rank_scores(
            torch.tensor([[0.0, 1e-5]]),
            torch.tensor([[10.0, 0.01]]),
        )
        self.assertEqual(membership.tolist(), [[True, False]])
        self.assertLess(scores[0, 0], scores[0, 1])

    def test_invalid_inner_radius_is_rejected(self):
        from src.system1.entailment_cones import (
            cone_half_aperture,
            validate_cone_parameters,
        )

        with self.assertRaises(ValueError):
            validate_cone_parameters(0.2, 0.1)
        with self.assertRaises(ValueError):
            cone_half_aperture(torch.tensor([[0.05, 0.0]]), 0.1, 0.1)

    def test_cone_energy_has_finite_gradients(self):
        from src.system1.entailment_cones import cone_energy

        for premise_value, theorem_value in (
            ([[0.3, 0.1]], [[0.6, 0.2]]),
            ([[0.3, 0.0]], [[0.6, 0.0]]),
            ([[0.6, 0.0]], [[0.3, 0.0]]),
        ):
            with self.subTest(premise=premise_value, theorem=theorem_value):
                premise = torch.tensor(premise_value, requires_grad=True)
                theorem = torch.tensor(theorem_value, requires_grad=True)
                loss = cone_energy(premise, theorem).sum()
                loss.backward()
                self.assertTrue(torch.isfinite(premise.grad).all())
                self.assertTrue(torch.isfinite(theorem.grad).all())

    def test_corrected_encoder_learns_node_specific_radii(self):
        from src.system1.corrected_cone_model import CorrectedConeEncoder

        torch.manual_seed(3)
        model = CorrectedConeEncoder(input_dim=4, output_dim=2)
        features = torch.randn(4, 4)
        messages = torch.tensor([[0, 1, 2], [1, 2, 3]])
        embeddings = model(features, messages)
        radii = embeddings.norm(dim=-1)
        self.assertTrue(torch.all(radii >= 0.1))
        self.assertTrue(torch.all(radii <= 0.95))
        self.assertGreater(
            float((radii.max() - radii.min()).detach()), 1e-6
        )

    def test_directional_training_step_is_finite(self):
        from src.system1.corrected_cone_model import CorrectedConeEncoder
        from src.system1.entailment_cones import cone_energy

        torch.manual_seed(9)
        model = CorrectedConeEncoder(input_dim=4, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        features = torch.randn(6, 4)
        messages = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
        )
        premise = torch.tensor([0, 1, 2, 3])
        theorem = torch.tensor([1, 2, 3, 4])
        negative = torch.tensor([5, 5, 0, 0])
        embeddings = model(features, messages)
        positive = cone_energy(embeddings[premise], embeddings[theorem]).mean()
        negative_loss = torch.relu(
            0.2 - cone_energy(embeddings[negative], embeddings[theorem])
        ).mean()
        radial = torch.relu(
            embeddings[premise].norm(dim=-1)
            - embeddings[theorem].norm(dim=-1)
            + 0.01
        ).mean()
        loss = positive + negative_loss + 0.2 * radial
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_inverse_retrieval_ranks_containing_premise_before_near_outside(self):
        from rebuttal.diagnose_corrected_cones import rank_candidates

        candidates = torch.tensor(
            [
                [0.30, 0.00],  # valid general premise
                [0.69, 0.10],  # close to query but outside its narrow cone
                [0.70, 0.00],  # query node itself, explicitly masked
            ]
        )
        queries = candidates[2:3]
        _, ranked, contained = rank_candidates(
            queries=queries,
            candidates=candidates,
            query_indices=torch.tensor([2]),
            arm="corrected_inverse",
            top_k=2,
            candidate_chunk_size=2,
            cone_k=0.1,
            epsilon=0.1,
        )
        self.assertEqual(ranked[0, 0].item(), 0)
        self.assertEqual(contained.item(), 1)


if __name__ == "__main__":
    unittest.main()
