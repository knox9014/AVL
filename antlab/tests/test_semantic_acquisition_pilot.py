import unittest
import torch
from antlab.semantic_acquisition_pilot import new_receiver,train_receiver,fingerprint


class AcquisitionPilotTests(unittest.TestCase):
    def test_fresh_same_width_initialization(self):
        a=new_receiver(16,401); b=new_receiver(16,401)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
        self.assertFalse(any(x.data_ptr()==y.data_ptr() for x,y in zip(a.parameters(),b.parameters())))

    def test_training_uses_detached_features_and_records_zero_step(self):
        torch.set_num_threads(2)
        z=torch.tensor([[-.8,0],[.8,0],[0,.8],[0,-.8]],dtype=torch.float32)
        target=torch.arange(4)
        evaluations={s:z.clone() for s in (44,55,66)}
        initial=new_receiver(2,401)
        model,record=train_receiver(z,target,evaluations,target,44,401,lambda:None)
        self.assertNotEqual(fingerprint(initial),fingerprint(model))
        self.assertEqual([r['step'] for r in record['curve']],[0,1,5,10,20,50,100])
        self.assertEqual(record['curve'][-1]['examples_seen'],400)
        self.assertEqual(set(record['curve'][-1]['senders']),{'44','55','66'})
        with self.assertRaises(ValueError): train_receiver(z.requires_grad_(),target,evaluations,target,44,401,lambda:None)


if __name__=='__main__': unittest.main()

