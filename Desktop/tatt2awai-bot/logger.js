// In logger.js
const winston = require('winston');
const path = require('path');

// Create logs directory if it doesn't exist
const fs = require('fs');
if (!fs.existsSync('logs')) {
    fs.mkdirSync('logs');
}

const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    defaultMeta: { service: 'tatt2awai-bot' },
    transports: [
        // Write all logs with importance level of 'error' or less to error.log
        new winston.transports.File({ 
            filename: path.join('logs', 'error.log'), 
            level: 'error',
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.json()
            )
        }),
        // Write all logs with importance level of 'info' or less to combined.log
        new winston.transports.File({ 
            filename: path.join('logs', 'combined.log'),
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.json()
            )
        }),
        // Write to console with colors
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple(),
                winston.format.printf(info => {
                    const {
                        timestamp, level, message, ...args
                    } = info;
                    
                    const ts = timestamp.slice(0, 19).replace('T', ' ');
                    return `${ts} ${level}: ${message} ${Object.keys(args).length ? JSON.stringify(args, null, 2) : ''}`;
                })
            )
        })
    ]
});

module.exports = logger;
