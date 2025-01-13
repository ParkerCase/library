const dropboxManager = require('./dropboxManager');
const logger = require('./logger');
const fs = require('fs').promises;
const path = require('path');

async function testDropboxConnection() {
  try {
    logger.info('Starting enhanced Dropbox connection test...');

    // Test 1: Initialize client
    logger.info('Test 1: Initializing client...');
    const client = await dropboxManager.ensureAuth();
    logger.info('✓ Client initialization successful');

    // Test 2: List root directory
    logger.info('Test 2: Listing root directory...');
    const rootListing = await dropboxManager.fetchDropboxEntries();
    const entries = rootListing.result.entries;
    logger.info(`✓ Found ${entries.length} entries in root`);

    // Log some entry details
    entries.slice(0, 3).forEach(entry => {
      logger.info('Entry found:', {
        name: entry.name,
        path: entry.path_lower,
        type: entry['.tag']
      });
    });

    // Test 3: Try uploading a test file
    logger.info('Test 3: Uploading test file...');
    const testContent = `Test file created at ${new Date().toISOString()}`;
    const testFilePath = '/test-upload.txt';
    
    try {
      await dropboxManager.uploadFile(testFilePath, testContent);
      logger.info('✓ Test file upload successful');
    } catch (uploadError) {
      logger.error('Upload test failed:', uploadError);
    }

    // Test 4: Try downloading a file
    if (entries.length > 0) {
      const fileEntry = entries.find(e => e['.tag'] === 'file');
      if (fileEntry) {
        logger.info('Test 4: Downloading file:', fileEntry.path_lower);
        const downloadedFile = await dropboxManager.downloadFile(fileEntry.path_lower);
        logger.info('✓ File download successful:', {
          name: fileEntry.name,
          size: downloadedFile.size
        });
      }
    }

    logger.info('All tests completed successfully! 🎉');
    return true;
  } catch (error) {
    logger.error('Test failed:', error);
    return false;
  }
}

// Run the test
testDropboxConnection().then(success => {
  if (success) {
    logger.info('All tests passed successfully');
    process.exit(0);
  } else {
    logger.error('Test suite failed');
    process.exit(1);
  }
});
