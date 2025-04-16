import express from 'express';
import { 
  verifyTrial, 
  auditTrial 
} from '../controllers/regulatorController.js';
import { protect } from '../middlewares/authMiddleware.js';
import { authorize } from '../middlewares/roleMiddleware.js';

const router = express.Router();

router.post('/verify/:trialId', protect, authorize('regulator'), verifyTrial);
router.get('/audit/:trialId', protect, authorize('regulator'), auditTrial);

export default router;