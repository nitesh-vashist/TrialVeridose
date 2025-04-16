import { logger } from '../utils/logger.js';

export const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      logger.error('User object not found in request');
      return res.status(500).json({
        success: false,
        error: 'Server error - user not authenticated properly'
      });
    }

    if (!roles.includes(req.user.role)) {
      logger.error(`User role ${req.user.role} not authorized to access this route`);
      return res.status(403).json({
        success: false,
        error: `User role ${req.user.role} not authorized to access this route`
      });
    }
    
    next();
  };
};