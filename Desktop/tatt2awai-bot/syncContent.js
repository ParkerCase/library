const { createClient } = require('@supabase/supabase-js');
const dropboxManager = require('./dropboxManager');
const { addJob } = require('./queues');
const logger = require('./logger');
const path = require('path');
const fs = require('fs');

// Initialize Supabase
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

// Ensure uploads directory exists
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Add this new function
async function updateImageSequences(fileInfo) {
  try {
    const { path: filePath, id } = fileInfo;
    const dirPath = path.dirname(filePath);
    
    logger.info('Updating image sequences:', { filePath, dirPath });

    // Find other images in the same directory
    const files = await dropboxManager.fetchDropboxEntries(dirPath);
    const imageFiles = files.result.entries
      .filter(e => ['.jpg', '.jpeg', '.png'].some(ext => 
        e.path_lower.endsWith(ext)
      ))
      .sort((a, b) => a.path_lower.localeCompare(b.path_lower));

    logger.info('Found image files:', { count: imageFiles.length });

    // Create sequence entries
    for (let i = 0; i < imageFiles.length; i++) {
      const { data, error } = await supabase
        .from('image_sequences')
        .upsert({
          image_id: imageFiles[i].id,
          prev_image: i > 0 ? imageFiles[i-1].id : null,
          next_image: i < imageFiles.length-1 ? imageFiles[i+1].id : null,
          sequence_order: i,
          sequence_group: dirPath,
          sequence_metadata: {
            directory: dirPath,
            total_images: imageFiles.length,
            file_name: path.basename(imageFiles[i].path_lower),
            position: `${i + 1} of ${imageFiles.length}`
          }
        });

      if (error) {
        throw error;
      }
    }

    return imageFiles.length;
  } catch (error) {
    logger.error('Error updating image sequences:', error);
    throw error;
  }
}

// Modify your processDropboxFile function to include sequence handling
async function processDropboxFile(entry, dropboxClient) {
  let tempPath = null;
  try {
    const response = await dropboxClient.filesDownload({ path: entry.path_lower });
    if (!response.result) {
      throw new Error('File download failed');
    }

    tempPath = path.join(UPLOADS_DIR, `temp_${Date.now()}_${path.basename(entry.path_lower)}`);
    fs.writeFileSync(tempPath, response.result.fileBinary);

    // Add sequence handling for images
    if (['.jpg', '.jpeg', '.png'].some(ext => entry.path_lower.endsWith(ext))) {
      await updateImageSequences({
        path: entry.path_lower,
        id: entry.id
      });
    }

const queueResult = await addJob({ 
  entry,
path: entry.path_lower,  // Add this explicitly
  metadata: {
    id: entry.id,
    name: entry.name,
    size: entry.size,
    timestamp: new Date().toISOString()
  }
});

    return queueResult;

  } catch (error) {
    logger.error('File processing error:', {
      path: entry.path_lower,
      error: error.message,
      stack: error.stack
    });
    throw error;
  } finally {
    if (tempPath && fs.existsSync(tempPath)) {
      try {
        fs.unlinkSync(tempPath);
      } catch (cleanupError) {
        logger.warn('Failed to clean up temporary file:', {
          path: tempPath,
          error: cleanupError.message
        });
      }
    }
  }
}

// In syncContent.js
async function syncContent({ forceFull = false, maxEntries = null, fileTypes = null } = {}) {
  try {
    logger.info('Starting Dropbox content sync...', { 
      forceFull, 
      maxEntries, 
      fileTypes 
    });
    
    // Get authenticated client
    const dropboxClient = await dropboxManager.ensureAuth();
    
    // Track sync progress
    let stats = {
      total: 0,
      files: 0,
      folders: 0,
      processed: 0,
      skipped: 0,
      errors: 0,
      paths: [],
      startTime: Date.now()
    };

    // Get initial file listing
    let response = await dropboxManager.fetchDropboxEntries();
    if (!response?.result?.entries) {
      throw new Error('No entries returned from Dropbox');
    }

    let entries = response.result.entries;
    logger.info(`Found ${entries.length} entries`);

    // Process entries
    for (const entry of entries) {
      try {
        // Skip folders
        if (entry['.tag'] === 'folder') {
          stats.folders++;
          stats.total++;
          continue;
        }

        // Validate file type
        const fileExt = path.extname(entry.path_lower).toLowerCase().replace('.', '');
        if (fileTypes && !fileTypes.includes(fileExt)) {
          stats.skipped++;
          stats.total++;
          continue;
        }

        stats.files++;
        stats.total++;

        if (maxEntries && stats.processed >= maxEntries) {
          logger.info('Reached maximum entries limit', { maxEntries });
          break;
        }

 // Debug log the entry data
                logger.info('Processing entry:', {
                    path: entry.path_lower,
                    name: entry.name,
                    tag: entry['.tag'],
                    hasPath: !!entry.path_lower
                });

        // Add to processing queue
const jobData = {
                    entry: {
                        ...entry,
                        path_lower: entry.path_lower // Ensure this is explicitly set
                    }
                };

                logger.info('Adding job with data:', jobData);

                await addJob(jobData);
       
 stats.processed++;
        stats.paths.push(entry.path_lower);

        // Log progress periodically
        if (stats.processed % 10 === 0) {
          logger.info('Sync progress:', {
            processed: stats.processed,
            total: stats.files,
            skipped: stats.skipped,
            errors: stats.errors
          });
        }

      } catch (entryError) {
        stats.errors++;
        logger.error('Entry processing error:', {
          path: entry.path_lower,
          error: entryError.message
        });
      }
    }

    // Calculate final stats
    stats.duration = Date.now() - stats.startTime;
    stats.successRate = ((stats.processed - stats.errors) / stats.processed * 100).toFixed(2);

    logger.info('Sync process completed', stats);
    return stats;

  } catch (error) {
    logger.error('Sync process failed:', error);
    throw error;
  }
}

module.exports = {
  syncContent,
  processDropboxFile, // Exported for testing
updateImageSequences
};
