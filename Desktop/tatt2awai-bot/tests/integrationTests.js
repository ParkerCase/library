// tests/integrationTests.js

const dropboxManager = require('../dropboxManager');
const { processUserMessage } = require('../openaiClient');
const logger = require('../logger');

async function runIntegrationTests() {
    try {
        console.log('Starting integration tests...\n');

        // Test 1: Basic Dropbox Connection
        console.log('Test 1: Testing Dropbox Connection');
        const dropboxStatus = await dropboxManager.validateConnection();
        console.log('Dropbox connected:', dropboxStatus);
        console.log('------------------------\n');

        // Test 2: Image Search with Tattoo Filter
        console.log('Test 2: Testing Image Search');
        const searchResult = await dropboxManager.searchFiles('tattoo', 'image');
        console.log('Tattoo-related images found:', searchResult.length);
        if (searchResult.length > 0) {
            console.log('First result path:', searchResult[0].path_lower);
            
            // Try to process the first found image
            const imageAnalysis = await dropboxManager.processDropboxImage(searchResult[0].path_lower);
            console.log('Image analysis completed:', {
                isTattoo: imageAnalysis?.analysis?.tattooFeatures?.isTattoo,
                skinDetected: imageAnalysis?.analysis?.tattooFeatures?.skinDetected
            });
        }
        console.log('------------------------\n');

        // Test 3: Assistant Integration
        console.log('Test 3: Testing Assistant Integration');
        const testUserId = 'test-user-' + Date.now();
        
        // If we found a tattoo image, use its path in the query
        let searchQuery = "find tattoo images";
        if (searchResult.length > 0) {
            searchQuery = `analyze the tattoo at ${searchResult[0].path_lower}`;
        }
        
        const response = await processUserMessage(testUserId, searchQuery);
        console.log('Assistant response:', {
            threadId: response.thread_id,
            hasContent: !!response.content,
            messageContent: response.content[0]?.text?.value
        });
        console.log('------------------------\n');

        console.log('All tests completed successfully!');
    } catch (error) {
        console.error('Test failed:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack
        });
        process.exit(1);
    }
}

// Add command line execution
if (require.main === module) {
    runIntegrationTests()
        .then(() => process.exit(0))
        .catch(error => {
            console.error('Tests failed:', error);
            process.exit(1);
        });
}

module.exports = { runIntegrationTests };
