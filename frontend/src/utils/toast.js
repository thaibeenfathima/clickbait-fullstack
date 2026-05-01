import toast from 'react-hot-toast';

// Custom toast configurations
const toastConfig = {
    success: {
        duration: 3000,
        style: {
            background: '#10b981',
            color: '#fff',
            fontWeight: '500',
            borderRadius: '12px',
            padding: '16px',
            boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)',
        },
        iconTheme: {
            primary: '#fff',
            secondary: '#10b981',
        },
    },
    error: {
        duration: 4000,
        style: {
            background: '#ef4444',
            color: '#fff',
            fontWeight: '500',
            borderRadius: '12px',
            padding: '16px',
            boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)',
        },
        iconTheme: {
            primary: '#fff',
            secondary: '#ef4444',
        },
    },
    loading: {
        style: {
            background: '#0ea5e9',
            color: '#fff',
            fontWeight: '500',
            borderRadius: '12px',
            padding: '16px',
            boxShadow: '0 4px 12px rgba(14, 165, 233, 0.3)',
        },
    },
};

// Toast helper functions
export const showSuccess = (message) => {
    return toast.success(message, toastConfig.success);
};

export const showError = (message) => {
    return toast.error(message, toastConfig.error);
};

export const showLoading = (message) => {
    return toast.loading(message, toastConfig.loading);
};

export const showInfo = (message) => {
    return toast(message, {
        duration: 3000,
        style: {
            background: '#8b5cf6',
            color: '#fff',
            fontWeight: '500',
            borderRadius: '12px',
            padding: '16px',
            boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)',
        },
        iconTheme: {
            primary: '#fff',
            secondary: '#8b5cf6',
        },
    });
};

export const dismissToast = (toastId) => {
    toast.dismiss(toastId);
};

export const showPromise = (promise, messages) => {
    return toast.promise(
        promise,
        {
            loading: messages.loading || 'Processing...',
            success: messages.success || 'Success!',
            error: messages.error || 'Something went wrong',
        },
        {
            success: toastConfig.success,
            error: toastConfig.error,
            loading: toastConfig.loading,
        }
    );
};

export default toast;
