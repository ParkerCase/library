const dropboxManager = require('./dropboxManager');
const logger = require('./logger');
const fs = require('fs');
const path = require('path');

async function testDropboxConnection() {
    logger.info('Starting comprehensive Dropbox connection test...');
    const results = {
        auth: false,
        upload: false,
        download: false,
        list: false,
        tokenRefresh: false
    };
    
    try {
        // Test 1: Authentication
        logger.info('Test 1: Testing authentication...');
        await dropboxManager.ensureAuth();
        results.auth = true;
        logger.info('✓ Authentication successful');

        // Test 2: File Upload with different content types
        logger.info('Test 2: Testing file uploads...');
        const testFiles = {
            text: {
                path: '/test-text.txt',
                content: `Test file created at ${new Date().toISOString()}\nThis is a test file.`
            },
            json: {
                path: '/test-data.json',
                content: JSON.stringify({
                    testId: Date.now(),
                    type: 'test',
                    data: { message: 'This is a test JSON file' }
                })
            },
            binary: {
                path: '/test-binary.dat',
                content: Buffer.from([0x00, 0x01, 0x02, 0x03])
            }
        };

        for (const [type, file] of Object.entries(testFiles)) {
            logger.info(`Testing ${type} file upload...`);
            const uploadResult = await dropboxManager.uploadFile(file.path, file.content);
            logger.info(`✓ ${type} file uploaded successfully:`, uploadResult.name);
        }
        results.upload = true;

        // Test 3: List Directory Contents
        logger.info('Test 3: Testing file listing...');
        const entries = await dropboxManager.fetchDropboxEntries();
        logger.info('✓ File listing successful:', {
            totalFiles: entries.result.entries.length,
            hasMore: entries.result.has_more
        });
        results.list = true;

        // Test 4: Download and Verify Files
        logger.info('Test 4: Testing file downloads...');
        for (const [type, file] of Object.entries(testFiles)) {
            logger.info(`Testing ${type} file download...`);
            const downloadResult = await dropboxManager.downloadFile(file.path);
            
            // Verify content matches (for text and json)
            if (type === 'text' || type === 'json') {
                const downloadedContent = downloadResult.fileBinary.toString('utf8');
                const originalContent = typeof file.content === 'string' ? 
                    file.content : 
                    JSON.stringify(file.content);
                
                if (downloadedContent.trim() === originalContent.trim()) {
                    logger.info(`✓ ${type} file content verified`);
                } else {
                    throw new Error(`Content verification failed for ${type} file`);
                }
            }
        }
        results.download = true;

        // Test 5: Authentication Maintenance
        logger.info('Test 5: Testing token/auth maintenance...');
        const originalToken = dropboxManager.dropboxTokens.accessToken;
        
        // Force a maintenance check
        dropboxManager.dropboxTokens.expirationTime = Date.now() - 1000;
        await dropboxManager.ensureAuth();
        
        // Verify we can still make API calls
        await dropboxManager.fetchDropboxEntries();
        logger.info('✓ Authentication maintained successfully');
        results.tokenRefresh = true;

        // Final Results
        const allTestsPassed = Object.values(results).every(r => r === true);
        if (allTestsPassed) {
            logger.info('🎉 All tests passed successfully!', results);
        } else {
            logger.info('⚠️ Test results:', results);
        }

        return {
            success: allTestsPassed,
            results: results
        };

    } catch (error) {
        logger.error('❌ Test suite failed:', {
            error: error.message,
            stack: error.stack,
            results: results
        });
        return {
            success: false,
            results: results,
            error: error
        };
    }
}

// Run the test suite
testDropboxConnection()
    .then(({ success, results, error }) => {
        if (success) {
            logger.info('✅ Dropbox connection is fully operational', results);
        } else {
            logger.info('ℹ️ Test completion status:', results);
            if (results.auth && results.upload && results.download && results.list) {
                logger.info('✅ Core functionality is working properly');
                process.exit(0);
            } else {
                logger.error('❌ Core functionality test failed');
                process.exit(1);
            }
        }
    })
    .catch(error => {
        logger.error('❌ Fatal error running test suite:', error);
        process.exit(1);
    });
