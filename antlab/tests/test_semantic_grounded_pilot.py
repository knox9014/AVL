import unittest
import torch
from antlab.semantic_grounded_pilot import VectorNet, wire, objective
from antlab.semantic_grounded_codec import anchors


class GroundedPilotTests(unittest.TestCase):
    def test_same_initialization_isolates_grounding(self):
        torch.manual_seed(44); first=VectorNet(10,16,0)
        torch.manual_seed(44); second=VectorNet(10,16,1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first.parameters(),second.parameters())))

    def test_gradient_and_wire(self):
        for width,contract in [(16,0),(16,1),(2,1)]:
            model=VectorNet(10,width,contract)
            vectors=model.encode(torch.tensor([[1,2,3],[2,3,0]]),torch.tensor([3,2]))
            loss=objective(model,vectors,torch.tensor([0,1]))
            loss.backward()
            self.assertTrue(model.projection.weight.grad.abs().sum()>0)
            self.assertTrue(model.receiver[0].weight.grad.abs().sum()>0)
            restored=wire(vectors,contract)
            self.assertTrue(torch.equal(restored,vectors.detach()))
            self.assertEqual(model.receive(restored).shape,(2,4))
            with self.assertRaises(ValueError): model.receive(torch.zeros(1,width+1))

    def test_anchor_loss_is_dimension_comparable(self):
        for width in (2,16):
            model=VectorNet(10,width,1)
            points=torch.tensor(anchors(width),dtype=torch.float32)
            target=torch.arange(4)
            expected=torch.nn.functional.cross_entropy(model.receive(points),target)
            self.assertTrue(torch.allclose(objective(model,points,target),expected))


if __name__=='__main__': unittest.main()

