const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const logger = require('./logger');
require('dotenv').config();

// Check required environment variables
const requiredEnvVars = {
    DROPBOX_CLIENT_ID: process.env.DROPBOX_CLIENT_ID,
    DROPBOX_CLIENT_SECRET: process.env.DROPBOX_CLIENT_SECRET,
    DROPBOX_SELECT_USER: process.env.DROPBOX_SELECT_USER
};

// Validate environment variables
Object.entries(requiredEnvVars).forEach(([key, value]) => {
    if (!value) {
        logger.error(`Missing required environment variable: ${key}`);
        process.exit(1);
    }
    logger.info(`Found ${key}: ${value.substring(0, 5)}...`);
});

const PORT = 4002;
const SERVER_IP = '147.182.247.128';
const TOKEN_FILE = path.join(__dirname, '.token-store.json');

// SSL certificate paths
const SSL_KEY_PATH = process.env.SSL_KEY_PATH || '/root/tatt2awai-bot/server.key';
const SSL_CERT_PATH = process.env.SSL_CERT_PATH || '/root/tatt2awai-bot/server.cert';

// Required scopes for team access
const REQUIRED_SCOPES = [
    'team_data.member',
    'files.content.read',
    'files.content.write',
    'team_data.team_space'
].join(' ');

const app = express();

// Add request logging middleware
app.use((req, res, next) => {
    logger.info(`Incoming ${req.method} request to ${req.url} from ${req.ip}`);
    next();
});

// Basic error handler
app.use((err, req, res, next) => {
    logger.error('Express error:', err);
    res.status(500).send('Internal Server Error');
});

// Health check route
app.get('/health', (req, res) => {
    logger.info('Health check request received');
    res.send('OK');
});

// Root route
app.get('/', (req, res) => {
    logger.info('Serving root page');
    res.send(`<a href="/auth">Click here to start Dropbox OAuth flow</a>`);
});

// Auth route
app.get('/auth', (req, res) => {
    // Log environment variables (sanitized)
    logger.info('Using client_id:', process.env.DROPBOX_CLIENT_ID?.substring(0, 5) + '...');
    
    const redirectUri = `https://${SERVER_IP}:${PORT}/callback`;
    const authUrl = `https://www.dropbox.com/oauth2/authorize` +
        `?client_id=${process.env.DROPBOX_CLIENT_ID}` +
        `&response_type=code` +
        `&token_access_type=offline` +
        `&scope=${encodeURIComponent(REQUIRED_SCOPES)}` +
        `&redirect_uri=${encodeURIComponent(redirectUri)}`;

    logger.info('Generated auth URL:', authUrl);
    res.redirect(authUrl);
});

// Callback route
app.get('/callback', async (req, res) => {
    const { code, error } = req.query;
    
    if (error) {
        logger.error('OAuth error:', error);
        return res.status(400).send(`OAuth error: ${error}`);
    }

    if (!code) {
        logger.error('No authorization code received');
        return res.status(400).send('Authorization code missing');
    }

    try {
        const redirectUri = `https://${SERVER_IP}:${PORT}/callback`;
        logger.info('Exchanging code for token with redirect URI:', redirectUri);
        
        const tokenResponse = await axios.post('https://api.dropboxapi.com/oauth2/token', null, {
            params: {
                code,
                grant_type: 'authorization_code',
                client_id: process.env.DROPBOX_CLIENT_ID,
                client_secret: process.env.DROPBOX_CLIENT_SECRET,
                redirect_uri: redirectUri
            }
        });

        const tokenData = {
            accessToken: tokenResponse.data.access_token,
            refreshToken: tokenResponse.data.refresh_token,
            expirationTime: Date.now() + (tokenResponse.data.expires_in * 1000)
        };

        fs.writeFileSync(TOKEN_FILE, JSON.stringify(tokenData, null, 2));
        logger.info('Successfully saved token data');
        
        res.send(`
            <h1>Authorization Successful!</h1>
            <p>Tokens have been saved to ${TOKEN_FILE}</p>
            <p>Please update your .env file with the following values:</p>
            <pre>
DROPBOX_ACCESS_TOKEN=${tokenData.accessToken}
DROPBOX_REFRESH_TOKEN=${tokenData.refreshToken}
            </pre>
            <p>You can now close this window and restart your application.</p>
        `);

    } catch (error) {
        logger.error('Error exchanging code for token:', error.response?.data || error);
        res.status(500).send('Error obtaining access token. Check server logs.');
    }
});

// Create HTTPS server
try {
    const httpsOptions = {
        key: fs.readFileSync(SSL_KEY_PATH),
        cert: fs.readFileSync(SSL_CERT_PATH)
    };

    const server = https.createServer(httpsOptions, app);
    
    server.listen(PORT, '0.0.0.0', () => {
        logger.info(`OAuth HTTPS server running on https://${SERVER_IP}:${PORT}`);
        logger.info(`Visit https://${SERVER_IP}:${PORT}/auth to start the OAuth flow`);
    });

} catch (error) {
    logger.error('Error starting HTTPS server:', error);
    process.exit(1);
}
