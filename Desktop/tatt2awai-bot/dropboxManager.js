const fs = require('fs');
const path = require('path');
const { Dropbox } = require('dropbox');
const logger = require('./logger');
require('dotenv').config();
const searchUtils = require('./searchUtils');
const fetch = require('node-fetch');
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_KEY
);

const TOKEN_FILE = path.join(__dirname, '.token-store.json');

let dropboxManagerInstance = null;
const AUTH_CACHE_DURATION = 15 * 60 * 1000; // 15 minutes
const AUTH_TOKEN_EXPIRY = 3600000; // 1 hour in milliseconds
const AUTH_REFRESH_BUFFER = 300000; // 5 minutes buffer before expiry
const AUTH_TIMEOUT = 45000;  // 45 seconds
const MAX_RETRIES = 5;       // Increase retries



const RATE_LIMIT_SETTINGS = {
    MAX_RETRIES: 3,
    RETRY_DELAY: 5000,
    MAX_CONCURRENT: 10
};

const TIMEOUTS = {
    DOWNLOAD: 120000,  // 120 seconds for downloads
    AUTH: 30000,       // 30 seconds for auth
    OPERATION: 60000   // 60 seconds for other operations
};

class DropboxManager {
    constructor() {
        // Ensure singleton pattern
        if (dropboxManagerInstance) {
            return dropboxManagerInstance;
        }

        this.dropboxTokens = {
            accessToken: null,
            refreshToken: null,
            expirationTime: null
        };
        this.dropboxClient = null;
        this.lastAuthCheck = null;
        this.authToken = null;
        this.lastSyncTime = null;
        this.lastTokenRefresh = 0;
        this.isRefreshing = false;
        
        // Rate limiting settings
        this.requestQueue = [];
        this.isProcessing = false;
        this.rateLimitDelay = 1000;
        this.retryDelay = 5000;
        this.maxRetries = 3;
        this.concurrentRequests = 0;
        this.maxConcurrentRequests = 10;
        this.MIN_REFRESH_INTERVAL = 60000;

        // Initialize
        this.loadTokens();
        this.startQueueProcessor();

        if (this.dropboxTokens.accessToken) {
            logger.info('Dropbox token status:', {
                hasAccessToken: !!this.dropboxTokens.accessToken,
                hasRefreshToken: !!this.dropboxTokens.refreshToken,
                tokenExpiry: this.dropboxTokens.expirationTime,
                isExpired: Date.now() > this.dropboxTokens.expirationTime
            });
        } else {
            logger.error('No Dropbox tokens loaded');
        }
        
        dropboxManagerInstance = this;
    }

    // Get singleton instance
    static getInstance() {
        if (!dropboxManagerInstance) {
            dropboxManagerInstance = new DropboxManager();
        }
        return dropboxManagerInstance;
    }

    // Add timeout handling
    async withTimeout(promise, ms, errorMessage) {
        let timeoutHandle;
        const timeoutPromise = new Promise((_, reject) => {
            timeoutHandle = setTimeout(() => {
                reject(new Error(`Timeout: ${errorMessage}`));
            }, ms);
        });

        try {
            const result = await Promise.race([promise, timeoutPromise]);
            clearTimeout(timeoutHandle);
            return result;
        } catch (error) {
            clearTimeout(timeoutHandle);
            throw error;
        }
    }
async initializeClient() {
        try {
            const config = {
                accessToken: this.dropboxTokens.accessToken
            };
            
            this.dropboxClient = new Dropbox(config);
            
            if (this.dropboxClient.customHeaders === undefined) {
                this.dropboxClient.customHeaders = {};
            }
            
            if (process.env.DROPBOX_SELECT_USER) {
                this.dropboxClient.customHeaders['Dropbox-API-Select-User'] = process.env.DROPBOX_SELECT_USER;
            }
            
            logger.info('Initialized Dropbox client with unified access configuration');
            
            // Verify connection
            const response = await this.makeRateLimitedRequest(async () => {
                return await this.dropboxClient.filesListFolder({
                    path: ''
                });
            });

            return true;
        } catch (error) {
            logger.error('Error initializing Dropbox client:', error);
            throw error;
        }
    }

    async startQueueProcessor() {
        while (true) {
            try {
                if (this.requestQueue.length > 0 && this.concurrentRequests < this.maxConcurrentRequests) {
                    const request = this.requestQueue.shift();
                    await this.processRequest(request);
                }
                await new Promise(resolve => setTimeout(resolve, 100));
            } catch (error) {
                logger.error('Error in queue processor:', error);
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
    }

    async processRequest(request) {
        this.concurrentRequests++;
        try {
            await request();
        } catch (error) {
            logger.error('Error processing request:', error);
        } finally {
            this.concurrentRequests--;
            await new Promise(resolve => setTimeout(resolve, this.rateLimitDelay));
        }
    }

async makeRateLimitedRequest(operation) {
    const baseDelay = this.retryDelay;
    return new Promise((resolve, reject) => {
        const attempt = async (retryCount = 0) => {
            try {
                const result = await this.withTimeout(
                    operation(),
                    45000,  // 45 second timeout
                    'Operation timed out'
                );
                resolve(result);
            } catch (error) {
                try {
                    // Enhanced error handling
                    if (error.status === 429) {
                        const retryAfter = parseInt(error.headers?.['retry-after']) || 60;
                        await new Promise(resolve => 
                            setTimeout(resolve, retryAfter * 1000)
                        );
                        return attempt(retryCount);
                    }

                    const action = await this.handleError(error, { retryCount });
                    if (action === 'retry' && retryCount < this.maxRetries) {
                        const delay = baseDelay * Math.pow(2, retryCount) * 
                            (1 + Math.random() * 0.1); // Add jitter
                        await new Promise(resolve => setTimeout(resolve, delay));
                        return attempt(retryCount + 1);
                    }
                } catch (handlerError) {
                    reject(handlerError);
                }
            }
        };

        this.requestQueue.push(() => attempt());
    });
}

    loadTokens() {
        try {
            if (fs.existsSync(TOKEN_FILE)) {
                const tokens = JSON.parse(fs.readFileSync(TOKEN_FILE, 'utf8'));
                this.dropboxTokens = tokens;
                logger.info('Loaded existing Dropbox tokens');
            } else if (process.env.DROPBOX_ACCESS_TOKEN) {
                this.dropboxTokens = {
                    accessToken: process.env.DROPBOX_ACCESS_TOKEN,
                    refreshToken: process.env.DROPBOX_REFRESH_TOKEN,
                    expirationTime: Date.now() + (3600 * 1000)
                };
                this.saveTokens();
                logger.info('Initialized tokens from environment variables');
            }
        } catch (error) {
            logger.error('Error loading tokens:', error);
        }
    }

    saveTokens() {
        try {
            fs.writeFileSync(TOKEN_FILE, JSON.stringify(this.dropboxTokens, null, 2));
            logger.info('Tokens saved successfully');
        } catch (error) {
            logger.error('Error saving tokens:', error);
        }
    }
async refreshAccessToken() {
        try {
            // Prevent multiple simultaneous refreshes
            if (this.isRefreshing) {
                return false;
            }

            this.isRefreshing = true;
            logger.info('Starting token refresh process...');

            // Add rate limit backoff
            const timeSinceLastRefresh = Date.now() - this.lastTokenRefresh;
            if (timeSinceLastRefresh < this.MIN_REFRESH_INTERVAL) {
                await new Promise(resolve => 
                    setTimeout(resolve, this.MIN_REFRESH_INTERVAL - timeSinceLastRefresh)
                );
            }

            const response = await this.withTimeout(
                fetch('https://api.dropboxapi.com/oauth2/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    body: new URLSearchParams({
                        grant_type: 'refresh_token',
                        refresh_token: this.dropboxTokens.refreshToken,
                        client_id: process.env.DROPBOX_CLIENT_ID,
                        client_secret: process.env.DROPBOX_CLIENT_SECRET
                    })
                }),
                30000,  // 30 second timeout
                'Token refresh timed out'
            );

            if (!response.ok) {
                throw new Error(`Token refresh failed with status ${response.status}`);
            }

            const data = await response.json();
            
            this.dropboxTokens = {
                accessToken: data.access_token,
                refreshToken: data.refresh_token || this.dropboxTokens.refreshToken,
                expirationTime: Date.now() + ((data.expires_in || 14400) * 1000)
            };

            this.lastTokenRefresh = Date.now();
            this.saveTokens();
            await this.initializeClient();

            return true;

        } catch (error) {
            logger.error('Error refreshing token:', {
                service: "tatt2awai-bot",
                error: error.message
            });
            throw error;
        } finally {
            this.isRefreshing = false;
        }
    }

    async ensureAuth() {
        try {
            return await this.withTimeout(
                this._ensureAuth(),
AUTH_TIMEOUT,
                'Authentication timed out'
            );
 } catch (error) {
        // Enhanced error logging
        logger.error('Auth error with timeout:', {
            service: "tatt2awai-bot",
            error: error.message,
            retryable: error.status === 401 || error.code === 'ERR_SSL_HTTP_REQUEST',
            timestamp: new Date().toISOString()
        });

        // Special handling for rate limits
        if (error.status === 429) {
            const retryAfter = parseInt(error.headers?.['retry-after']) || 60;
            logger.warn(`Rate limited, waiting ${retryAfter} seconds`);
            await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
            return this.ensureAuth();
        }

        throw error;
    }
}

async _ensureAuth() {
    try {
        const now = Date.now();
        
        // Enhanced cache check
        if (this.authToken && 
            this.lastAuthCheck && 
            (now - this.lastAuthCheck) < AUTH_CACHE_DURATION &&
            this.dropboxTokens.expirationTime > (now + 5 * 60 * 1000)) {
            return true;
        }

        if (!this.dropboxClient) {
            await this.initializeClient();
        }
        
        // More aggressive token refresh
        const needsRefresh = !this.dropboxTokens.expirationTime || 
            Date.now() > (this.dropboxTokens.expirationTime - (15 * 60 * 1000)); // 15 min buffer
                
        if (needsRefresh) {
            await this.refreshAccessToken();
        }

        // Enhanced retry logic
        let retries = MAX_RETRIES;
        let lastError;
        
        while (retries > 0) {
            try {
                await this.makeRateLimitedRequest(async () => {
                    const response = await this.dropboxClient.filesListFolder({
                        path: '',
                        limit: 1
                    });
                    return response;
                });
                
                this.authToken = this.dropboxTokens.accessToken;
                this.lastAuthCheck = now;
                
                logger.info('Successfully authenticated with Dropbox', {
                    service: "tatt2awai-bot",
                    remainingValidity: Math.round((this.dropboxTokens.expirationTime - now) / 1000) + 's'
                });
                return true;
                
            } catch (error) {
                lastError = error;
                
                if (error?.status === 401 && retries > 1) {
                    logger.info('Token expired during test, refreshing...', { retries });
                    await this.refreshAccessToken();
                } else if (error?.code === 'ERR_SSL_HTTP_REQUEST' && retries > 1) {
                    logger.info('SSL error encountered, retrying...', { retries });
                    await new Promise(resolve => setTimeout(resolve, 2000 * (MAX_RETRIES - retries + 1)));
                } else if (retries === 1) {
                    throw error;
                }
                retries--;
                await new Promise(resolve => 
                    setTimeout(resolve, this.retryDelay * Math.pow(2, MAX_RETRIES - retries))
                );
            }
        }
        throw lastError || new Error('Failed to authenticate after retries');
    } catch (error) {
        logger.error('Error ensuring authentication:', {
            error: error.message,
            code: error.code,
            status: error.status,
            timestamp: new Date().toISOString()
        });
        
        if (error?.status === 401 && !error._isRetry) {
            logger.info('Final attempt: refreshing token');
            await this.refreshAccessToken();
            error._isRetry = true;
            return await this.ensureAuth();
        }
        return false;
    }
}

async downloadFile(filePath) {
    await this.ensureAuth();
    try {
        // Increased timeout and added retry logic while keeping original structure
        return await this.withTimeout(
            this._downloadFileWithRetry(filePath),
            TIMEOUTS.DOWNLOAD * 2, // Double the timeout
            'File download timed out'
        );
    } catch (error) {
        await this.trackFailedOperation(filePath, error);
        throw error;
    }
}

// Add this new helper method
async _downloadFileWithRetry(filePath, attempt = 1, maxRetries = 3) {
    try {
        return await this._downloadFile(filePath);
    } catch (error) {
        if (attempt >= maxRetries) {
            throw error;
        }

        // Add delay between retries using exponential backoff
        await new Promise(resolve => 
            setTimeout(resolve, 2000 * Math.pow(2, attempt - 1))
        );

        // Refresh token if auth error
        if (error.status === 401) {
            await this.ensureAuth(true); // Force token refresh
        }

        return this._downloadFileWithRetry(filePath, attempt + 1, maxRetries);
    }
}

async _downloadFile(filePath) {
    const normalizedPath = filePath.startsWith('/') ? filePath : `/${filePath}`;
    
    const response = await this.makeRateLimitedRequest(async () => {
        const downloadResponse = await this.dropboxClient.filesDownload({ 
            path: normalizedPath,
            // Add small delay between requests to prevent rate limiting
            ...(await new Promise(resolve => setTimeout(resolve, 100)))
        });
        if (!downloadResponse?.result) {
            throw new Error('Invalid download response');
        }
        return downloadResponse;
    });

    let fileBinary;
    if (Buffer.isBuffer(response.result.fileBinary)) {
        fileBinary = response.result.fileBinary;
    } else if (response.result.fileBlob) {
        const arrayBuffer = await response.result.fileBlob.arrayBuffer();
        fileBinary = Buffer.from(arrayBuffer);
    } else if (typeof response.result === 'object' && response.result.fileBinary) {
        fileBinary = Buffer.from(response.result.fileBinary);
    } else {
        fileBinary = Buffer.from(response.result);
    }

    // Add validation
    if (!fileBinary || fileBinary.length === 0) {
        throw new Error('Empty file binary received');
    }

    return {
        result: {
            path: normalizedPath,
            fileBinary,
            metadata: {
                path: normalizedPath,
                name: response.result.name,
                size: response.result.size,
                serverModified: response.result.server_modified,
                clientModified: response.result.client_modified
            }
        }
    };
}

    async uploadFile(filePath, content) {
        await this.ensureAuth();
        return this.withTimeout(
            this.makeRateLimitedRequest(async () => {
                try {
                    const normalizedPath = filePath.startsWith('/') ? filePath : `/${filePath}`;
                    const response = await this.dropboxClient.filesUpload({
                        path: normalizedPath,
                        contents: content,
                        mode: { '.tag': 'overwrite' }
                    });
                    return response;
                } catch (error) {
                    if (error?.status === 401) {
                        await this.refreshAccessToken();
                        return await this.uploadFile(filePath, content);
                    }
                    throw error;
                }
            }),
            60000,  // 60 second timeout
            'File upload timed out'
        );
    }

    async fetchDropboxEntries(path = '') {
        await this.ensureAuth();
        try {
            const initialResponse = await this.withTimeout(
                this.makeRateLimitedRequest(() =>
                    this.dropboxClient.filesListFolder({
                        path,
                        recursive: true,
                        include_mounted_folders: true,
                        include_non_downloadable_files: true,
                        limit: 2000,
                    })
                ),
                30000,  // 30 second timeout
                'Listing folder timed out'
            );

            let entries = initialResponse.result.entries || [];
            let hasMore = initialResponse.result.has_more;
            let cursor = initialResponse.result.cursor;

            while (hasMore) {
                const continuedResponse = await this.makeRateLimitedRequest(() =>
                    this.dropboxClient.filesListFolderContinue({ cursor })
                );
                entries = entries.concat(continuedResponse.result.entries || []);
                hasMore = continuedResponse.result.has_more;
                cursor = continuedResponse.result.cursor;
            }

            const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
            const imageEntries = entries.filter(entry =>
                imageExtensions.some(ext => entry.path_lower.endsWith(ext))
            );

            const groupedEntries = {};
            imageEntries.forEach(entry => {
                const dirPath = entry['.tag'] === 'folder' ? 
                    entry.path_lower : 
                    entry.path_lower.substring(0, entry.path_lower.lastIndexOf('/'));
                
                if (!groupedEntries[dirPath]) {
                    groupedEntries[dirPath] = [];
                }
                groupedEntries[dirPath].push(entry);
            });

            return {
                result: {
                    entries: imageEntries,
                    groupedEntries,
                }
            };
        } catch (error) {
            if (error?.status === 401) {
                await this.refreshAccessToken();
                return await this.fetchDropboxEntries(path);
            }
            throw error;
        }
    }

    async handleError(error, context) {
        logger.error('Dropbox operation failed:', {
            service: "tatt2awai-bot",
            context,
            error: error.message,
            status: error?.status
        });

        if (error?.status === 401 || error?.code === 'ExpiredAuthError') {
            await this.refreshAccessToken();
            return 'retry';
        }

        if (error?.status === 429) {
            const retryAfter = parseInt(error.headers?.['retry-after'] || '60', 10);
            await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
            return 'retry';
        }

        throw error;
    }

    async trackFailedOperation(path, error) {
        try {
            await supabase
                .from('failed_operations')
                .upsert({
                    path,
                    last_error: error.message,
                    error_count: 1,
                    last_attempt: new Date().toISOString()
                });
        } catch (dbError) {
            logger.error('Error tracking failed operation:', {
                service: "tatt2awai-bot",
                path,
                error: dbError.message
            });
        }
    }
async searchFiles(query, type = 'all') {
        await this.ensureAuth();
        logger.info('Searching files with query:', { 
            service: "tatt2awai-bot",
            query, 
            type 
        });

        try {
            const entries = await this.fetchDropboxEntries('');
            let files = entries.result.entries;

            logger.info('Total files in Dropbox:', { 
                service: "tatt2awai-bot",
                count: files.length 
            });

            if (type === 'image') {
                files = files.filter(file => {
                    const ext = path.extname(file.path_lower).toLowerCase();
                    return ['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext);
                });
                logger.info('Filtered image files:', { 
                    service: "tatt2awai-bot",
                    count: files.length 
                });
            }

            const searchResults = files.filter(file => {
                const searchString = `${file.path_lower} ${file.name}`.toLowerCase();
                return searchString.includes(query.toLowerCase());
            });

            logger.info('Search results:', {
                service: "tatt2awai-bot",
                query,
                resultsCount: searchResults.length
            });

            return searchResults;
        } catch (error) {
            logger.error('Error searching files:', {
                service: "tatt2awai-bot",
                error: error.message
            });
            throw error;
        }
    }

    async searchForImage(imageData, query) {
        await this.ensureAuth();
        try {
            const entries = await this.fetchDropboxEntries('');
            const imageFiles = entries.result.entries.filter(entry =>
                ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext =>
                    entry.path_lower.endsWith(ext)
                )
            );

            logger.info(`Found ${imageFiles.length} image files to search through`, {
                service: "tatt2awai-bot"
            });

            const matches = [];
            for (const file of imageFiles) {
                try {
                    const fileData = await this.downloadFile(file.path_lower);
                    if (fileData?.result?.fileBinary) {
                        matches.push({
                            path: file.path_lower,
                            data: fileData.result.fileBinary,
                            metadata: {
                                name: file.name,
                                size: file.size,
                                modified: file.server_modified
                            }
                        });
                    }
                } catch (error) {
                    logger.error(`Error processing file ${file.path_lower}:`, {
                        service: "tatt2awai-bot",
                        error: error.message
                    });
                }
            }

            return matches;
        } catch (error) {
            logger.error('Error searching for image:', {
                service: "tatt2awai-bot",
                error: error.message
            });
            throw error;
        }
    }

    async validateConnection() {
        try {
            await this.ensureAuth();
            const response = await this.withTimeout(
                this.makeRateLimitedRequest(async () => {
                    return await this.dropboxClient.filesListFolder({
                        path: '',
                        limit: 1
                    });
                }),
                30000,
                'Connection validation timed out'
            );
            return !!response;
        } catch (error) {
            logger.error('Connection validation failed:', {
                service: "tatt2awai-bot",
                error: error.message
            });
            return false;
        }
    }
}

// Add before module.exports
const dropboxEnhancements = {
    smartSync: {
        // ... your existing smartSync implementations stay the same
    }
};

// Add enhancement to the DropboxManager prototype
Object.assign(DropboxManager.prototype, dropboxEnhancements);

// Create and export singleton instance
const dropboxManager = new DropboxManager();
module.exports = dropboxManager;
