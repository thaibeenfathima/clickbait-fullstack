export default function LoadingSkeleton({
    type = 'text',
    className = '',
    count = 1,
    width = 'w-full',
    height = 'h-4'
}) {
    const skeletons = Array.from({ length: count });

    const getSkeletonClass = () => {
        switch (type) {
            case 'text':
                return `skeleton rounded ${height} ${width} ${className}`;
            case 'circle':
                return `skeleton rounded-full ${width} ${height} ${className}`;
            case 'card':
                return `skeleton rounded-lg w-full h-48 ${className}`;
            case 'avatar':
                return `skeleton rounded-full w-12 h-12 ${className}`;
            default:
                return `skeleton rounded ${height} ${width} ${className}`;
        }
    };

    return (
        <div className="space-y-3">
            {skeletons.map((_, index) => (
                <div key={index} className={getSkeletonClass()} />
            ))}
        </div>
    );
}
