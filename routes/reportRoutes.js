import express from 'express';
import { 
  submitPatientReport, 
  getTrialReports, 
  finalizeTrialReports, 
  getFinalReports 
} from '../controllers/reportController.js';
import { protect } from '../middlewares/authMiddleware.js';
import { authorize } from '../middlewares/roleMiddleware.js';

const router = express.Router();

router.post('/submit', protect, authorize('hospital'), submitPatientReport);
router.get('/trial/:trialId', protect, getTrialReports);
router.post('/finalize/:trialId', protect, authorize('hospital'), finalizeTrialReports);
router.get('/final/:trialId', protect, authorize('manufacturer', 'regulator'), getFinalReports);

export default router;