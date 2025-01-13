const path = require('path');
const fs = require('fs').promises;
const pLimit = require('p-limit');
const logger = require('./logger');
const dropboxManager = require('./dropboxManager');
const EnhancedSignatureGenerator = require('./enhanced-signature-generator');
const EnhancedSignatureStore = require('./enhanced-store');

// PART 1: State Management Constants
const SYSTEM_STATE = {
    SIGNATURE_FILE: path.join(__dirname, 'image-signatures.json'),
    VALIDATION_STATE_FILE: path.join(__dirname, 'signature-state.json'),
    CACHE_DURATION: 24 * 60 * 60 * 1000, // 24 hours
    BATCH_SIZE: 100,
    CONCURRENT_LIMIT: 10
};

// PART 2: Initialization Sequence
async function initializeSystem() {
    try {
        // Step 1: Load validation state
        const validationState = await loadValidationState();
        
        // Step 2: Load existing signatures
        const existingSignatures = await loadExistingSignatures();
        
        // Step 3: Validate against current Dropbox state
        const dropboxEntries = await dropboxManager.fetchDropboxEntries('');
        if (!dropboxEntries?.result?.entries) {
            throw new Error('Failed to fetch Dropbox entries');
        }

        // Step 4: Process any missing signatures
        const missingEntries = findMissingEntries(dropboxEntries.result.entries, existingSignatures);
        
        // Step 5: Generate missing signatures
        if (missingEntries.length > 0) {
            await processNewEntries(missingEntries);
        }

        return true;
    } catch (error) {
        logger.error('System initialization failed:', error);
        throw error;
    }
}

// PART 3: State Loading Functions
async function loadValidationState() {
    try {
        const stateData = await fs.readFile(SYSTEM_STATE.VALIDATION_STATE_FILE, 'utf8');
        return JSON.parse(stateData);
    } catch (error) {
        logger.warn('No validation state found, creating new');
        return { processedFiles: [] };
    }
}

async function loadExistingSignatures() {
    try {
        const signatureData = await fs.readFile(SYSTEM_STATE.SIGNATURE_FILE, 'utf8');
        const signatures = JSON.parse(signatureData);
        logger.info(`Loaded ${Object.keys(signatures).length} existing signatures`);
        return signatures;
    } catch (error) {
        logger.warn('No existing signatures found, creating new');
        return {};
    }
}

// PART 4: Processing Functions
function findMissingEntries(allEntries, existingSignatures) {
    return allEntries.filter(entry => {
        const isImage = ['.jpg', '.jpeg', '.png', '.webp'].some(ext => 
            entry.path_lower.endsWith(ext)
        );
        return isImage && !existingSignatures[entry.path_lower];
    });
}

async function processNewEntries(entries) {
    const limit = pLimit(SYSTEM_STATE.CONCURRENT_LIMIT);
    let processed = 0;

    // Process in batches
    for (let i = 0; i < entries.length; i += SYSTEM_STATE.BATCH_SIZE) {
        const batch = entries.slice(i, i + SYSTEM_STATE.BATCH_SIZE);
        
        await Promise.all(batch.map(entry => limit(async () => {
            try {
                const fileData = await dropboxManager.downloadFile(entry.path_lower);
                if (!fileData?.result?.fileBinary) return;

                const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                    fileData.result.fileBinary :
                    Buffer.from(fileData.result.fileBinary);

                const signature = await EnhancedSignatureGenerator.generateSignature(buffer);
                await saveSignature(entry.path_lower, signature);
                
                processed++;
                if (processed % 100 === 0) {
                    logger.info(`Processed ${processed}/${entries.length} new signatures`);
                }
            } catch (error) {
                logger.error(`Failed to process ${entry.path_lower}:`, error);
            }
        })));
    }
}

// PART 5: State Saving Functions
async function saveSignature(path, signature) {
    const signatures = await loadExistingSignatures();
    signatures[path] = signature;
    await fs.writeFile(
        SYSTEM_STATE.SIGNATURE_FILE,
        JSON.stringify(signatures, null, 2)
    );
}

// Export the system
module.exports = {
    initializeSystem,
    loadValidationState,
    loadExistingSignatures,
    findMissingEntries,
    processNewEntries,
    saveSignature,
    SYSTEM_STATE
};
