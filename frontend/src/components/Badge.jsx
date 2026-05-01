export default function Badge({ label, children, variant = 'primary', className = '' }) {
  const variants = {
    primary: 'bg-blue-500/20 text-blue-300 border border-blue-500/30',
    success: 'bg-green-500/20 text-green-300 border border-green-500/30',
    danger: 'bg-red-500/20 text-red-300 border border-red-500/30',
    warning: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30',
    neutral: 'bg-gray-500/20 text-gray-300 border border-gray-500/30',
  };

  return (
    <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium backdrop-blur-sm ${variants[variant]} ${className}`}>
      {label || children}
    </span>
  );
}
