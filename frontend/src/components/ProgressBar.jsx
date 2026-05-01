import { motion } from 'framer-motion';

export default function ProgressBar({
    progress = 0,
    showPercentage = true,
    className = '',
    color = 'blue',
    size = 'medium'
}) {
    const colorClasses = {
        blue: 'bg-blue-500',
        green: 'bg-green-500',
        purple: 'bg-purple-500',
        red: 'bg-red-500',
    };

    const sizeClasses = {
        small: 'h-1',
        medium: 'h-2',
        large: 'h-3',
    };

    const clampedProgress = Math.min(100, Math.max(0, progress));

    return (
        <div className={`w-full ${className}`}>
            {showPercentage && (
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-300">Progress</span>
                    <span className="text-sm font-semibold text-white">
                        {Math.round(clampedProgress)}%
                    </span>
                </div>
            )}
            <div className={`w-full bg-gray-700/50 rounded-full overflow-hidden ${sizeClasses[size]}`}>
                <motion.div
                    className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full`}
                    initial={{ width: 0 }}
                    animate={{ width: `${clampedProgress}%` }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                />
            </div>
        </div>
    );
}
