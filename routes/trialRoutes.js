import express from 'express';
import { 
  initiateTrial, 
  getTrials, 
  getTrial, 
  respondToTrial 
} from '../controllers/trialController.js';
import { protect } from '../middlewares/authMiddleware.js';
import { authorize } from '../middlewares/roleMiddleware.js';

const router = express.Router();

router.post('/initiate', protect, authorize('manufacturer'), initiateTrial);
router.get('/', protect, getTrials);
router.get('/:id', protect, getTrial);
router.post('/:id/response', protect, authorize('hospital'), respondToTrial);

export default router;