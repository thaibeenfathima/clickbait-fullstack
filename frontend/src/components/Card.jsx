export default function Card({ children, className = '' }) {
  return (
    <div className={`glass-dark rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 ${className}`}>
      {children}
    </div>
  );
}
