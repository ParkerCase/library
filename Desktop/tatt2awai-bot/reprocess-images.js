const dropboxManager = require('./dropboxManager');
const imageProcessor = require('./imageProcessor');
const logger = require('./logger');
const path = require('path');
const fs = require('fs');

// Ensure uploads directory exists
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

async function reprocessAllImages() {
    try {
        // Ensure Dropbox auth
        await dropboxManager.ensureAuth();
        
        // Get all files
        const entries = await dropboxManager.fetchDropboxEntries('');
        const imageFiles = entries.result.entries.filter(entry =>
            ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext =>
                entry.path_lower.endsWith(ext)
            )
        );

        logger.info(`Found ${imageFiles.length} images to reprocess`);

        for (const file of imageFiles) {
            let tempPath = null;
            
            try {
                // Download the file
                const fileData = await dropboxManager.downloadFile(file.path_lower);
                if (!fileData?.result?.fileBinary) {
                    logger.error(`No file data received for ${file.path_lower}`);
                    continue;
                }

                // Create temp file
                tempPath = path.join(UPLOADS_DIR, `temp_${Date.now()}_${path.basename(file.path_lower)}`);
                
                // Write the file data
                fs.writeFileSync(tempPath, fileData.result.fileBinary);
                
                // Verify file was written
                if (!fs.existsSync(tempPath)) {
                    throw new Error(`Failed to write temporary file: ${tempPath}`);
                }

                logger.info(`Reprocessing ${file.path_lower}`);
                
                // Process the image
                const analysis = await imageProcessor.processImage({ path: tempPath });
                
                // Log success
                logger.info(`Successfully processed ${file.path_lower}`, {
                    hasAnalysis: !!analysis,
                    tattooFeatures: !!analysis?.tattooFeatures
                });

            } catch (error) {
                logger.error(`Error reprocessing ${file.path_lower}:`, error);
            } finally {
                // Clean up temp file
                if (tempPath && fs.existsSync(tempPath)) {
                    try {
                        fs.unlinkSync(tempPath);
                    } catch (cleanupError) {
                        logger.warn(`Failed to clean up temp file ${tempPath}:`, cleanupError);
                    }
                }
            }

            // Add a small delay between files to avoid overwhelming the system
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        logger.info('Reprocessing complete');

    } catch (error) {
        logger.error('Reprocessing failed:', error);
    }
}

// Add error handling for uncaught errors
process.on('uncaughtException', error => {
    logger.error('Uncaught exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', error => {
    logger.error('Unhandled rejection:', error);
    process.exit(1);
});

reprocessAllImages();
