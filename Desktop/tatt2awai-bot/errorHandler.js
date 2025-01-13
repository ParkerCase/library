class SignatureError extends Error {
    constructor(message, code, details = {}) {
        super(message);
        this.name = 'SignatureError';
        this.code = code;
        this.details = details;
        this.timestamp = new Date().toISOString();
    }
}

class ValidationError extends SignatureError {
    constructor(message, details) {
        super(message, 'VALIDATION_ERROR', details);
    }
}

class ProcessingError extends SignatureError {
    constructor(message, details) {
        super(message, 'PROCESSING_ERROR', details);
    }
}

module.exports = {
    SignatureError,
    ValidationError,
    ProcessingError,
    handleError: (error) => {
        console.error(`[${new Date().toISOString()}] ${error.name}: ${error.message}`);
        console.error('Details:', error.details);
        // Add error reporting/monitoring here
    }
};
