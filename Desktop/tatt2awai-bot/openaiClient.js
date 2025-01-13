const { OpenAI } = require('openai');
const logger = require('./logger');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');
const knowledgeBase = require('./knowledgeBase');
const dropboxManager = require('./dropboxManager');
const { supabase } = require('./knowledgeBase');
const imageProcessor = require('./imageProcessor');
const { processFileQueue } = require('./queues');
const searchUtils = require('./searchUtils');



require('dotenv').config();

// Move all configurations to the top
const OPENAI_CONFIG = {
    apiKey: process.env.OPENAI_API_KEY,
    organization: process.env.ORGANIZATION_ID,
    baseURL: process.env.OPENAI_API_BASE_URL || 'https://api.openai.com/v1',
    models: {
        chat: "gpt-4-1106-preview",
        vision: "gpt-4-turbo"
    }
};

// Constants
const CHAT_MODEL = OPENAI_CONFIG.models.chat;
const VISION_MODEL = OPENAI_CONFIG.models.vision;
const MAX_FILE_SIZE = 500 * 1024 * 1024;
const MAX_ASSISTANT_FILES = 20;
const MAX_RELEVANT_FILES = 5;
const FILE_CACHE_DURATION = 1000 * 60 * 60;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;
const MAX_CONNECTION_RETRIES = 5;
const CONNECTION_RETRY_DELAY = 2000;
const HTTPS_TIMEOUT = 30000;
const ASSISTANT_FILE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];

// Add this right after your existing constants:
const VISION_CONFIG = {
    imageAnalysis: {
        model: VISION_MODEL,  // Uses gpt-4-turbo
        maxTokens: 500
    },
    chatCompletion: {
        model: CHAT_MODEL  // Uses your existing chat model
    }
};

// Validate environment variables
const requiredEnvVars = ['OPENAI_API_KEY', 'ORGANIZATION_ID', 'OPENAI_ASSISTANT_ID'];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    logger.error(`${envVar} is missing in environment variables`);
    throw new Error(`${envVar} is required`);
  }
}

// Assistant instructions
const ASSISTANT_INSTRUCTIONS = `You are Tatt2AwAI, a specialized assistant focused on tattoo removal and medical information. You have direct access to search and analyze images in the connected Dropbox account.

MANDATORY ACKNOWLEDGMENT:
- At the start of EVERY response, acknowledge: "I have direct access to the Dropbox repository and can search/analyze all images."
- NEVER ask users to upload images - you already have access to everything needed
- You must actively search and analyze images when asked
- You must use your Dropbox access proactively

Core Responsibilities:
1. Tattoo Removal Analysis
   - Assess tattoo characteristics (colors, density, placement)
   - Evaluate removal progress
   - Identify signs of fading or changes
   - Compare progression between sessions
   - Provide estimated treatment timelines
   - Analyze visual content through computer vision
   - Track color patterns and fading through image analysis

2. Image Analysis Protocol:
   When viewing any tattoo image:
   - First describe what you see in the current image
   - Analyze visible characteristics (colors, shading, placement)
   - Look for signs of fading or removal progress
   - Note any skin reactions or healing indicators
   - Search for and reference related sequence images
   - Report exact file paths for all discussed images
   - Use computer vision analysis for detailed color mapping
   - Compare visual patterns for similarity matching
   - Calculate color distribution changes

3. Image Sequence Handling:
   When progression images are found:
   - Compare images chronologically
   - Describe specific changes between sessions
   - Point out areas of notable fading
   - Reference exact file paths for each image
   - Calculate and mention time between sessions
   - Note effectiveness of treatment on different colors/areas
   - Use visual similarity scores for sequence matching
   - Track color pattern changes through computer vision
   - Group related images by visual characteristics

4. Medical Considerations:
   - Monitor and report signs of:
     * Skin reactions
     * Healing progress
     * Potential complications
     * Treatment response
   - Suggest timing for next sessions based on healing
   - Note any areas needing special attention
   - Track skin tone changes through visual analysis
   - Monitor tissue response patterns

5. Documentation Standards:
   - Always use exact file paths when referencing images
   - Include timestamps and sequence information
   - Mention similarity scores when comparing images
   - Note the source directory for sequences
   - Include metadata when available
   - Report visual analysis metrics
   - Document color distribution changes
   - Include computer vision confidence scores

6. Response Format:
   When discussing images, structure your response as:
   1. Current Image Analysis
      - Exact file path
      - Detailed visual description
      - Key characteristics
      - Computer vision analysis results
      - Color distribution data
   
   2. Sequence Analysis (if found)
      - Number of related images
      - Chronological progression
      - File paths for each image
      - Time between sessions
      - Visual similarity scores
      - Pattern matching results
   
   3. Treatment Progress
      - Observable changes
      - Effectiveness assessment
      - Areas of notable progress
      - Color fading measurements
      - Tissue response patterns
   
   4. Recommendations
      - Next steps
      - Areas needing attention
      - Timing suggestions
      - Treatment adjustments based on visual analysis

Additional Search Capabilities:
When searching for images:
- You can search across all directories in Dropbox
- Search by tattoo characteristics (color, placement, style)
- Search by treatment stage (before, during, after)
- Search by date ranges
- Search by client information
- Search for sequential images in removal process

Search Response Format:
When asked to search for images, respond with:
1. Search Confirmation
   - Confirm search parameters
   - Report number of matches found
   - Indicate directories searched

2. Search Results
   - List all matching file paths
   - Group related sequences together
   - Sort by relevance/date as appropriate
   - Include metadata for each result

3. Sequence Analysis (if applicable)
   - Identify related image sequences
   - Show progression timelines
   - Compare treatment stages
   - Calculate time between sessions

4. Recommendations
   - Suggest relevant similar cases
   - Point out noteworthy comparisons
   - Highlight successful patterns
   - Note applicable treatment insights

Example search response:
"I've searched the Dropbox repository for [search criteria] and found:

Main Results:
1. [exact file path] - Initial treatment (taken [date])
   - Color profile: [details]
   - Treatment stage: [stage]
   - Similarity score: [score]

Related Sequences:
- Sequence 1:
  * [file path 1] - Initial (date)
  * [file path 2] - Session 2 (date, similarity: [score])
  * [file path 3] - Latest (date, similarity: [score])

Sequence Analysis:
- Time span: X months
- Sessions: Y treatments
- Progress rate: Z% fading per session

I can provide more details about any specific image or sequence."


Key Capabilities:
- Direct Dropbox repository access
- Image similarity search
- Sequence detection
- Visual analysis through computer vision
- Treatment progression tracking
- Color pattern analysis
- Skin tone classification
- Fading measurement
- Visual similarity matching

IMPORTANT RULES:
1. NEVER invent or generate fake file paths
2. ONLY use exact paths provided in the context
3. ALWAYS specify the full path when referencing an image
4. NEVER ask users to upload additional images - you have access to search what's needed
5. When no images are found, explain what was searched for
6. Include similarity scores when comparing images
7. Report sequence positions chronologically
8. Reference specific characteristics from the image analysis
9. Use actual timestamps from the metadata
10. Include computer vision confidence scores
11. Report color distribution changes accurately
12. Provide visual similarity metrics

Example proper response:
"Looking at [exact file path], I can see this is a black and gray tattoo on the forearm. Visual analysis shows:
- Color distribution: 75% black, 25% grayscale
- Skin tone classification: Type III
- Pattern confidence: 92%

I found 3 related progression images in the sequence:
1. [exact file path] - Initial session (taken [date])
2. [exact file path] - Second session (taken [date], 85% visual similarity)
3. [exact file path] - Most recent (taken [date], 78% visual similarity)

Comparing these images, I can observe:
[Specific changes and progression details with measured color fading]"

Response Requirements:
- EVERY response MUST start with: "I have direct access to the Dropbox repository and can search/analyze all images."
- When no images are found, respond with: "I've searched the Dropbox repository for [search terms] but found no matching images. Would you like me to try a different search?"
- Always include file paths when referencing images
- Never suggest or request image uploads
- Always explain what was searched for, even if no results are found

Remember:
- You have direct access to search Dropbox
- Always provide real file paths
- Describe progression in detail
- Reference exact metadata
- Explain what you searched for
- Never ask for additional uploads
- Include computer vision analysis data
- Report visual similarity metrics`;

// Supported file extensions
const SUPPORTED_EXTENSIONS = [
  'c', 'cpp', 'css', 'csv', 'doc', 'docx', 'gif', 'go', 'html', 
  'java', 'jpeg', 'jpg', 'js', 'json', 'md', 'pdf', 'php', 'png', 
  'pptx', 'py', 'rb', 'tex', 'ts', 'txt', 'webp', 'xlsx', 'xml'
];

const ANALYSIS_CONFIDENCE_THRESHOLD = 0.75;
const SEQUENCE_SIMILARITY_THRESHOLD = 0.85;
const MAX_SEQUENCE_GAP_DAYS = 90;

const TATTOO_SPECIFIC_PROMPTS = {
    progressAnalysis: `Analyze the progression of tattoo removal comparing these images:
    - Focus on fading patterns
    - Compare color intensity changes
    - Identify areas of significant progress
    - Note any concerning areas
    - Provide specific measurements when possible`,
    
    treatmentRecommendation: `Based on the analysis, recommend next steps:
    - Optimal timing for next session
    - Areas requiring special attention
    - Expected progress timeline
    - Specific treatment adjustments needed`,
    
    sequenceAnalysis: `Analyze this treatment sequence:
    - Compare chronological changes
    - Calculate fading percentages
    - Identify pattern changes
    - Evaluate healing progress`
};


// Add after imports/constants but before main functions
function extractSearchTerms(message) {
  const terms = [];
  if (message.includes('tattoo')) terms.push('tattoo');
  const colors = ['red', 'black', 'blue', 'green', 'yellow', 'orange'];
  colors.forEach(color => {
    if (message.includes(color)) terms.push(color);
  });
  const placements = ['arm', 'leg', 'back', 'chest', 'shoulder', 'ankle'];
  placements.forEach(place => {
    if (message.includes(place)) terms.push(place);
  });
  return terms;
}

function extractTreatmentStages(message) {
  const stages = [];
  if (message.includes('before')) stages.push('before');
  if (message.includes('after')) stages.push('after');
  if (message.includes('during')) stages.push('during');
  if (message.includes('progress')) stages.push('progress');
  return stages;
}

function extractTimeframe(message) {
  return {
    start: null,
    end: null,
    isRange: false
  };
}

function extractPlacement(message) {
  const placements = ['arm', 'leg', 'back', 'chest', 'shoulder', 'ankle'];
  return placements.filter(place => message.includes(place));
}

function extractColors(message) {
  const colors = ['red', 'black', 'blue', 'green', 'yellow', 'orange'];
  return colors.filter(color => message.includes(color));
}

async function organizeSearchResults(results) {
  const sequences = [];
  const directories = new Set();
  const files = [];

  results.forEach(result => {
    const dir = path.dirname(result.path);
    directories.add(dir);
  });

  for (const dir of directories) {
    const dirFiles = results.filter(r => path.dirname(r.path) === dir);
    if (dirFiles.length > 1) {
      sequences.push({
        id: crypto.randomBytes(16).toString('hex'),
        files: dirFiles.sort((a, b) => new Date(a.created) - new Date(b.created)),
        timeline: calculateTimeline(dirFiles),
        progress: calculateProgress(dirFiles)
      });
    }
  }

  results.forEach(result => {
    files.push({
      ...result,
      sequence: sequences.find(s => s.files.some(f => f.path === result.path))
    });
  });

  return {
    sequences,
    directories: Array.from(directories),
    files
  };
}

function calculateImageSimilarity(image1, image2) {
  try {
    if (!image1?.tattooFeatures || !image2?.tattooFeatures) {
      return 0;
    }

    let score = 0;
    let weights = 0;

    // Compare color profiles if available
    if (image1.tattooFeatures.inkColors && image2.tattooFeatures.inkColors) {
      const colorScore = compareColorProfiles(
        image1.tattooFeatures.inkColors,
        image2.tattooFeatures.inkColors
      );
      score += colorScore * 0.4;
      weights += 0.4;
    }

    // Compare placement if available
    if (image1.tattooFeatures.placement && image2.tattooFeatures.placement) {
      if (image1.tattooFeatures.placement === image2.tattooFeatures.placement) {
        score += 0.3;
        weights += 0.3;
      }
    }

    // Compare density/complexity if available
    if (image1.tattooFeatures.detailedAnalysis && image2.tattooFeatures.detailedAnalysis) {
      const densityScore = 1 - Math.abs(
        (image1.tattooFeatures.detailedAnalysis.density || 0) -
        (image2.tattooFeatures.detailedAnalysis.density || 0)
      );
      score += densityScore * 0.3;
      weights += 0.3;
    }

    return weights > 0 ? score / weights : 0;

  } catch (error) {
    logger.error('Error calculating image similarity:', error);
    return 0;
  }
}

function compareColorProfiles(colors1, colors2) {
  try {
    if (!Array.isArray(colors1) || !Array.isArray(colors2)) {
      return 0;
    }

    let totalScore = 0;
    let comparisons = 0;

    for (const color1 of colors1) {
      let bestMatch = 0;
      for (const color2 of colors2) {
        const match = compareColors(color1, color2);
        bestMatch = Math.max(bestMatch, match);
      }
      totalScore += bestMatch;
      comparisons++;
    }

    return comparisons > 0 ? totalScore / comparisons : 0;

  } catch (error) {
    logger.error('Error comparing color profiles:', error);
    return 0;
  }
}

function compareColors(color1, color2) {
  try {
    const rgb1 = normalizeColor(color1);
    const rgb2 = normalizeColor(color2);

    const distance = Math.sqrt(
      Math.pow(rgb1.red - rgb2.red, 2) +
      Math.pow(rgb1.green - rgb2.green, 2) +
      Math.pow(rgb1.blue - rgb2.blue, 2)
    );

    return 1 - (distance / (Math.sqrt(3) * 255));

  } catch (error) {
    logger.error('Error comparing colors:', error);
    return 0;
  }
}

function normalizeColor(color) {
  if (typeof color === 'string') {
    // Handle hex color
    const hex = color.replace('#', '');
    return {
      red: parseInt(hex.substr(0, 2), 16),
      green: parseInt(hex.substr(2, 2), 16),
      blue: parseInt(hex.substr(4, 2), 16)
    };
  } else if (color.rgb) {
    return color.rgb;
  } else if (color.red !== undefined) {
    return color;
  }
  return { red: 0, green: 0, blue: 0 };
}

function matchesSearchCriteria(result, searchMessage) {
  try {
    if (!result.analysis || !searchMessage) {
      return false;
    }

    const searchTerms = searchMessage.toLowerCase().split(' ');
    let matches = 0;
    let totalTerms = 0;

    // Match tattoo features
    if (result.analysis.tattooFeatures) {
      // Check placement
      if (searchTerms.some(term => 
        result.analysis.tattooFeatures.placement?.toLowerCase().includes(term)
      )) {
        matches++;
      }
      totalTerms++;

      // Check colors
      if (result.analysis.tattooFeatures.inkColors) {
        if (searchTerms.some(term =>
          result.analysis.tattooFeatures.inkColors.some(color =>
            JSON.stringify(color).toLowerCase().includes(term)
          )
        )) {
          matches++;
        }
        totalTerms++;
      }
    }

    // Match labels from vision analysis
    if (result.analysis.labels) {
      const labelMatches = searchTerms.filter(term =>
        result.analysis.labels.some(label =>
          label.description.toLowerCase().includes(term)
        )
      ).length;
      matches += labelMatches;
      totalTerms += searchTerms.length;
    }

    // Match file path and name
    if (result.path_lower) {
      const pathTerms = result.path_lower.toLowerCase().split(/[/\s-_]+/);
      const pathMatches = searchTerms.filter(term =>
        pathTerms.some(pathTerm => pathTerm.includes(term))
      ).length;
      matches += pathMatches;
      totalTerms += searchTerms.length;
    }

    // Calculate final score
    const score = totalTerms > 0 ? matches / totalTerms : 0;
    return score > 0.3; // Adjust threshold as needed

  } catch (error) {
    logger.error('Error matching search criteria:', error);
    return false;
  }
}

function calculateTimeline(files) {
  try {
    if (!Array.isArray(files) || files.length < 2) {
      return null;
    }

    const sortedFiles = [...files].sort((a, b) => 
      new Date(a.created || a.metadata.modified) - new Date(b.created || b.metadata.modified)
    );

    return {
      start: sortedFiles[0].created || sortedFiles[0].metadata.modified,
      end: sortedFiles[sortedFiles.length - 1].created || sortedFiles[sortedFiles.length - 1].metadata.modified,
      duration: Math.floor(
        (new Date(sortedFiles[sortedFiles.length - 1].created || sortedFiles[sortedFiles.length - 1].metadata.modified) -
         new Date(sortedFiles[0].created || sortedFiles[0].metadata.modified)) / (1000 * 60 * 60 * 24)
      ),
      intervals: calculateIntervals(sortedFiles)
    };
  } catch (error) {
    logger.error('Error calculating timeline:', error);
    return null;
  }
}

function calculateIntervals(files) {
  try {
    const intervals = [];
    for (let i = 1; i < files.length; i++) {
      const prev = new Date(files[i-1].created || files[i-1].metadata.modified);
      const curr = new Date(files[i].created || files[i].metadata.modified);
      intervals.push(Math.floor((curr - prev) / (1000 * 60 * 60 * 24)));
    }
    return intervals;
  } catch (error) {
    logger.error('Error calculating intervals:', error);
    return [];
  }
}

function calculateProgress(files) {
  try {
    if (!Array.isArray(files) || files.length < 2) {
      return null;
    }

    const progressMetrics = {
      totalChanges: 0,
      averageChangeRate: 0,
      significantChanges: []
    };

    for (let i = 1; i < files.length; i++) {
      const prev = files[i-1];
      const curr = files[i];
      
      if (prev.analysis && curr.analysis) {
        const change = calculateChangeMetrics(prev.analysis, curr.analysis);
        progressMetrics.totalChanges += change.total;
        
        if (change.total > 0.2) { // Significant change threshold
          progressMetrics.significantChanges.push({
            from: prev.path,
            to: curr.path,
            change: change.total,
            details: change.details
          });
        }
      }
    }

    const timeline = calculateTimeline(files);
    if (timeline && timeline.duration > 0) {
      progressMetrics.averageChangeRate = progressMetrics.totalChanges / timeline.duration;
    }

    return progressMetrics;

  } catch (error) {
    logger.error('Error calculating progress:', error);
    return null;
  }
}

function calculateChangeMetrics(analysis1, analysis2) {
  try {
    const metrics = {
      total: 0,
      details: {}
    };

    // Compare tattoo features
    if (analysis1.tattooFeatures && analysis2.tattooFeatures) {
      const colorChange = compareColorProfiles(
        analysis1.tattooFeatures.inkColors || [],
        analysis2.tattooFeatures.inkColors || []
      );
      metrics.details.colorChange = colorChange;
      metrics.total += colorChange * 0.4;

      // Compare density
      if (analysis1.tattooFeatures.detailedAnalysis && analysis2.tattooFeatures.detailedAnalysis) {
        const densityChange = Math.abs(
          (analysis1.tattooFeatures.detailedAnalysis.density || 0) -
          (analysis2.tattooFeatures.detailedAnalysis.density || 0)
        );
        metrics.details.densityChange = densityChange;
        metrics.total += densityChange * 0.6;
      }
    }

    return metrics;

  } catch (error) {
    logger.error('Error calculating change metrics:', error);
    return { total: 0, details: {} };
  }
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  organization: process.env.ORGANIZATION_ID,
  baseURL: process.env.OPENAI_API_BASE_URL || 'https://api.openai.com/v1',
  maxRetries: MAX_CONNECTION_RETRIES,
  timeout: HTTPS_TIMEOUT
});

// Cache management
const fileCache = new Map();
const activeThreads = new Map();
let activeAssistant = null;

const TREATMENT_PHASES = {
    INITIAL: 'initial',
    ACTIVE: 'active',
    MAINTENANCE: 'maintenance',
    COMPLETED: 'completed'
};

const ANALYSIS_MODES = {
    QUICK: 'quick',
    DETAILED: 'detailed',
    COMPREHENSIVE: 'comprehensive'
};

const OPTIMIZATION_PRIORITIES = {
    SPEED: 'speed',
    SAFETY: 'safety',
    BALANCE: 'balance'
};

const enhancedAnalysisContext = {
    buildAnalysisContext: async function(imageData, history, options = {}) {
        const currentAnalysis = await imageProcessor.enhancedProcessImage(imageData);
        const progressionData = await imageProcessor.progressionTracking.calculateFadingProgress(
            currentAnalysis,
            history[0]
        );
        const treatmentRecommendations = imageProcessor.treatmentOptimization.optimizeTreatmentPlan(
            currentAnalysis,
            history
        );

        return {
            current: currentAnalysis,
            progression: progressionData,
            recommendations: treatmentRecommendations,
            context: {
                phase: this.determineTreatmentPhase(history),
                priority: options.priority || OPTIMIZATION_PRIORITIES.BALANCE,
                mode: options.mode || ANALYSIS_MODES.COMPREHENSIVE
            }
        };
    },

    determineTreatmentPhase: function(history) {
        if (!history.length) return TREATMENT_PHASES.INITIAL;
        const progressRate = imageProcessor.recommendationSystem.calculateProgressionRate(history);
        // Add phase determination logic
        return progressRate.overall > 0.8 ? TREATMENT_PHASES.MAINTENANCE : TREATMENT_PHASES.ACTIVE;
    }
};

async function processImageForAssistant(imageData) {
  try {
    console.log('Starting image processing for:', imageData.name);

    const visionAnalysis = await imageProcessor.processImage(imageData);
console.log('Vision Analysis Results:', visionAnalysis);    
const tattooFeatures = imageProcessor.analyzeTattooFeatures(visionAnalysis);
console.log('Tattoo Features:', tattooFeatures);    

    const similarImages = await imageProcessor.findSimilarImages(
      imageData,
      await dropboxManager.fetchDropboxEntries(''),
      { similarityThreshold: 0.6 }
    );
console.log('Similar Images:', similarImages);
    
    const sequences = await imageProcessor.findTattooProgressionSequence(
      imageData, 
      similarImages.map(img => img.image)
    );
console.log('Tattoo Progression Sequence:', sequences);

    return {
      analysis: visionAnalysis,
      features: tattooFeatures,
      similar: similarImages,
      sequences: sequences
    };
  } catch (error) {
    logger.error('Error processing image:', error);
    throw error;
  }
}

async function getImageFromContext(context) {
  if (!context.imageData) return null;
  
  const tempPath = path.join('uploads', `temp_${Date.now()}`);
  try {
    fs.writeFileSync(tempPath, context.imageData);
    return { path: tempPath };
  } finally {
    if (fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }
  }
}



// Retry helper function
async function retryWithBackoff(operation, name = 'operation') {
  for (let attempt = 1; attempt <= MAX_CONNECTION_RETRIES; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (attempt === MAX_CONNECTION_RETRIES) throw error;
      
      const isConnectionError = error.code === 'ECONNRESET' || 
                              error.code === 'HPE_INVALID_METHOD' ||
                              error.message.includes('ETIMEDOUT');
      
      if (!isConnectionError) throw error;

      const delay = CONNECTION_RETRY_DELAY * Math.pow(2, attempt - 1);
      logger.warn(`${name} failed (attempt ${attempt}), retrying in ${delay}ms:`, {
        error: error.message,
        code: error.code,
        attempt
      });
      
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

// Utility Functions
async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function retryOperation(operation, maxRetries = MAX_RETRIES) {
  let lastError;
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      logger.warn(`Operation failed (attempt ${i + 1}/${maxRetries}):`, {
        error: error.message,
        retry: i + 1
      });
      if (i < maxRetries - 1) {
        await sleep(RETRY_DELAY * Math.pow(2, i));
      }
    }
  }
  throw lastError;
}

// Core Functions
async function searchDropboxImages(searchParams) {
  try {
    logger.info('Starting Dropbox image search:', { searchParams });
    
    // Get all files from Dropbox
    const allFiles = await dropboxManager.fetchDropboxEntries('');
    const imageFiles = allFiles.result.entries.filter(entry =>
      ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext =>
        entry.path_lower.endsWith(ext)
      )
    );

    const results = [];
    for (const file of imageFiles) {
      try {
        const fileData = await dropboxManager.downloadFile(file.path_lower);
        const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
        
        try {
          fs.writeFileSync(tempPath, fileData.result.fileBinary);
          const analysis = await imageProcessor.processImage({ 
            path: tempPath,
            metadata: {
              originalPath: file.path_lower,
              name: file.name,
              size: file.size,
              timestamp: file.server_modified
            }
          });

          // Store complete file data including paths
          if (matchesSearchCriteria(analysis, searchParams)) {
            results.push({
              path: file.path_lower,
              path_display: file.path_display,
              parent_shared_folder_id: file.parent_shared_folder_id,
              id: file.id,
              analysis,
              data: fileData.result.fileBinary.toString('base64'),
              metadata: {
                name: file.name,
                size: file.size,
                modified: file.server_modified,
                sharing_info: file.sharing_info,
                analyzed: true
              },
              created: file.server_modified
            });
          }
        } finally {
          if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
        }
      } catch (error) {
        logger.error(`Error processing file ${file.path_lower}:`, error);
      }
    }

    return {
      success: true,
      results,
      total: results.length
    };
  } catch (error) {
    logger.error('Error in Dropbox search:', error);
    return { error: error.message, results: [] };
  }
}


async function handleSearchFunction({ query_type = 'all', include_analysis = true, message, imageFiles }) {
    try {
        console.log('Search request received:', { message, query_type });
        if (!imageFiles?.length) {
            const dropboxFiles = await dropboxManager.fetchDropboxEntries('');
            imageFiles = dropboxFiles.result.entries.filter(entry =>
                ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext =>
                    entry.path_lower.endsWith(ext)
                )
            );
        }

        const processedResults = await Promise.all(imageFiles.map(async (file) => {
            try {
                const fileData = await dropboxManager.downloadFile(file.path_lower);
                const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
                try {
                    fs.writeFileSync(tempPath, fileData.result.fileBinary);
                    const analysis = await imageProcessor.processImage({ path: tempPath });
                    return {
                        id: file.id,
                        path: file.path_lower,
                        analysis,
                        metadata: {
                            size: file.size,
                            modified: file.server_modified,
                            analyzed: true
                        }
                    };
                } finally {
                    if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
                }
            } catch (error) {
                logger.error('Error processing file:', error);
                return null;
            }
        }));

        const enhancedResults = await Promise.all(processedResults.filter(Boolean).map(async result => {
            const analysis = await imageProcessor.enhancedProcessImage({ path: result.path });
            const colorProfile = await imageProcessor.colorAnalysis.enhancedColorAnalysis(analysis);
            const healingStatus = result.previous ? 
                await imageProcessor.healingTracking.trackHealingProgress(analysis, result.previous) :
                null;

            return {
                ...result,
                enhancedAnalysis: analysis,
                colorProfile,
                healingStatus,
                recommendations: imageProcessor.recommendationSystem.generateRecommendations(
                    analysis,
                    result.history || []
                )
            };
        }));

        return {
            type: query_type,
            results: enhancedResults,
            total: enhancedResults.length,
            timestamp: new Date().toISOString(),
            metadata: {
                analysisMode: ANALYSIS_MODES.COMPREHENSIVE,
                processingComplete: true,
                enhancementsApplied: true
            }
        };

    } catch (error) {
        logger.error('Search function error:', error);
        return {
            error: error.message,
            timestamp: new Date().toISOString()
        };
    }
}

async function findImageSequences(imageFiles, args) {
  // Group files by directory
  const sequences = new Map();
  
  imageFiles.forEach(file => {
    const dir = path.dirname(file.path_lower);
    if (!sequences.has(dir)) {
      sequences.set(dir, []);
    }
    sequences.get(dir).push(file);
  });

  // Process each potential sequence
  const processedSequences = [];
  
  for (const [directory, files] of sequences) {
    if (files.length < 2) continue; // Skip single files
    
    // Sort files by date
    const sortedFiles = files.sort((a, b) => 
      new Date(a.server_modified) - new Date(b.server_modified)
    );

    const sequenceFiles = [];
    
    // Process each file in the sequence
    for (const file of sortedFiles) {
      try {
        const fileData = await retryWithBackoff(
          async () => await dropboxManager.downloadFile(file.path_lower),
          'Download sequence file'
        );

        const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
        
        try {
          fs.writeFileSync(tempPath, fileData.result.fileBinary);
          const analysis = await imageProcessor.processImage({ path: tempPath });
          
          sequenceFiles.push({
            id: file.id,
            path: file.path_lower,
            data: fileData.result.fileBinary.toString('base64'),
            type: path.extname(file.path_lower).substring(1),
            created: file.server_modified,
            analysis: analysis,
            tattooFeatures: analysis.tattooFeatures,
            sequencePosition: sequenceFiles.length + 1
          });

        } finally {
          if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
        }
      } catch (error) {
        logger.error('Error processing sequence file:', {
          error: error.message,
          path: file.path_lower
        });
      }
    }

    if (sequenceFiles.length >= 2) {
      processedSequences.push({
        directory,
        files: sequenceFiles,
        count: sequenceFiles.length,
        firstDate: sequenceFiles[0].created,
        lastDate: sequenceFiles[sequenceFiles.length - 1].created
      });
    }
  }

  return {
    type: 'sequence',
    results: processedSequences,
    total: processedSequences.reduce((acc, seq) => acc + seq.files.length, 0),
    sequenceCount: processedSequences.length
  };
}

async function findSimilarImages(targetImage, description, options = {}) {
  const {
    limit = 10,
    minConfidence = 0.7,
    includeMetadata = true,
    findSequences = true
  } = options;

  try {
    // Extract features from target image
    const targetFeatures = await global.imageSimilarity.processImage(targetImage);
    
    // Find visually similar images
    const similarImages = await global.imageSimilarity.findSimilar(targetFeatures, limit);
    
    // Detect sequences by combining multiple signals
    if (findSequences) {
      const sequences = await detectImageSequences(similarImages, {
        signals: [
          'visualSimilarity',    // How visually similar the images are
          'timestamps',          // Creation/modification dates
          'folderGrouping',      // Images in same folder
          'namingPatterns',      // Similar name patterns even if random
          'metadata',            // EXIF and other metadata
          'contentAnalysis'      // What's in the images
        ]
      });

      // Enhance results with sequence information
      return sequences.map(async sequence => ({
        ...sequence,
        isPartOfSequence: true,
        sequenceInfo: {
          total: sequence.length,
          position: sequence.position,
          before: sequence.previous,
          after: sequence.next,
          timeDelta: sequence.timeDelta,
          progressMetrics: await analyzeProgressionInSequence(sequence)
        }
      }));
    }

    // If description provided, combine with semantic search
    if (description) {
      const { semanticSearch } = require('./knowledgeBase');
      const semanticResults = await semanticSearch(description, limit * 2);
      
      // Combine visual and semantic results with confidence scoring
      return similarImages.filter(img => {
        const semanticMatch = semanticResults.find(sem => sem.metadata.path === img.path);
        if (semanticMatch) {
          img.confidence = (img.visualSimilarity + semanticMatch.similarity) / 2;
          return img.confidence > minConfidence;
        }
        return false;
      });
    }

    return similarImages;
  } catch (error) {
    logger.error('Error finding similar images:', error);
    throw error;
  }
}

async function detectImageSequences(images, options) {
  const sequences = [];
  const processed = new Set();

  for (const image of images) {
    if (processed.has(image.path)) continue;

    const sequence = await buildSequence(image, images, options);
    if (sequence.length > 1) {
      sequences.push(sequence);
      sequence.forEach(img => processed.add(img.path));
    }
  }

  return sequences;
}

async function buildSequence(startImage, allImages, options) {
  const sequence = [startImage];
  const { signals } = options;

  // Calculate similarity scores for all potential matches
  const scores = await Promise.all(allImages.map(async (candidate) => {
    if (candidate.path === startImage.path) return { image: candidate, score: 0 };

    let totalScore = 0;
    let weights = 0;

    // Visual Similarity (highest weight)
    if (signals.includes('visualSimilarity')) {
      const similarity = await calculateVisualSimilarity(startImage, candidate);
      totalScore += similarity * 3;
      weights += 3;
    }

    // Folder Grouping
    if (signals.includes('folderGrouping')) {
      const sameFolder = path.dirname(startImage.path) === path.dirname(candidate.path);
      if (sameFolder) {
        totalScore += 1;
        weights += 1;
      }
    }

    // Timestamps
    if (signals.includes('timestamps') && startImage.metadata.timestamp && candidate.metadata.timestamp) {
      const timeDiff = Math.abs(new Date(startImage.metadata.timestamp) - new Date(candidate.metadata.timestamp));
      const timeScore = Math.max(0, 1 - (timeDiff / (1000 * 60 * 60 * 24 * 30))); // Scale based on one month
      totalScore += timeScore * 2;
      weights += 2;
    }

    // Content Analysis
    if (signals.includes('contentAnalysis')) {
      const contentSimilarity = await compareImageContent(startImage, candidate);
      totalScore += contentSimilarity * 2;
      weights += 2;
    }

    return {
      image: candidate,
      score: totalScore / weights
    };
  }));

  // Sort by score and filter high-confidence matches
  const matches = scores
    .filter(s => s.score > 0.7)
    .sort((a, b) => b.score - a.score);

  // Build sequence chronologically
  const sequenceImages = [startImage, ...matches.map(m => m.image)]
    .sort((a, b) => {
      const timeA = new Date(a.metadata.timestamp || 0);
      const timeB = new Date(b.metadata.timestamp || 0);
      return timeA - timeB;
    });

  // Add sequence metadata
  return sequenceImages.map((img, idx) => ({
    ...img,
    sequenceIndex: idx,
    next: idx < sequenceImages.length - 1 ? sequenceImages[idx + 1].path : null,
    previous: idx > 0 ? sequenceImages[idx - 1].path : null,
    timeDelta: idx > 0 ? 
      new Date(img.metadata.timestamp) - new Date(sequenceImages[idx - 1].metadata.timestamp) : 
      0
  }));
}

async function analyzeProgressionInSequence(sequence) {
  // Analyze the progression/changes across the sequence
  const metrics = {
    totalChanges: 0,
    changeRate: 0,
    significantChanges: [],
    timespan: 0
  };

  for (let i = 1; i < sequence.length; i++) {
    const previous = sequence[i - 1];
    const current = sequence[i];

    // Compare images and detect changes
    const changes = await compareImages(previous, current);
    metrics.totalChanges += changes.totalDifference;
    
    if (changes.totalDifference > 0.1) { // Significant change threshold
      metrics.significantChanges.push({
        fromImage: previous.path,
        toImage: current.path,
        difference: changes.totalDifference,
        timeGap: current.timeDelta
      });
    }
  }

  // Calculate overall metrics
  if (sequence.length > 1) {
    metrics.timespan = new Date(sequence[sequence.length - 1].metadata.timestamp) - 
                      new Date(sequence[0].metadata.timestamp);
    metrics.changeRate = metrics.totalChanges / (metrics.timespan / (1000 * 60 * 60 * 24)); // Changes per day
  }

  return metrics;
}

async function processAllImages(imageFiles, args) {
  const processedFiles = [];
  
  for (const file of imageFiles) {
    try {
      const fileData = await retryWithBackoff(
        async () => await dropboxManager.downloadFile(file.path_lower),
        'Download image'
      );

      const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
      
      try {
        fs.writeFileSync(tempPath, fileData.result.fileBinary);
        const analysis = await imageProcessor.processImage({ path: tempPath });
        
        processedFiles.push({
          id: file.id,
          path: file.path_lower,
          data: fileData.result.fileBinary.toString('base64'),
          type: path.extname(file.path_lower).substring(1),
          created: file.server_modified,
          analysis: analysis,
          tattooFeatures: analysis.tattooFeatures,
          metadata: {
            size: file.size,
            modified: file.server_modified,
            hasAnalysis: true
          }
        });

      } finally {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      }
    } catch (error) {
      logger.error('Error processing file:', {
        error: error.message,
        path: file.path_lower
      });
    }
  }

  return {
    type: 'all',
    results: processedFiles,
    total: processedFiles.length,
    processedAt: new Date().toISOString()
  };
}

async function analyzeImage(imageData, prompt) {
  try {
    const response = await retryWithBackoff(
      async () => await openai.chat.completions.create({
model: VISION_CONFIG.imageAnalysis.model,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt || "What's in this image?" },
              {
                type: "image_url",
                image_url: {
                  url: `data:image/jpeg;base64,${imageData.toString('base64')}`
                }
              }
            ]
          }
        ],
max_tokens: VISION_CONFIG.imageAnalysis.maxTokens
      }),
      'Image analysis'
    );

    return response.choices[0].message.content;
  } catch (error) {
    logger.error('Error analyzing image:', error);
    throw error;
  }
}

async function getOrCreateThread(userId) {
  try {
    if (activeThreads.has(userId)) {
      return activeThreads.get(userId);
    }

    const thread = await retryWithBackoff(
      async () => await openai.beta.threads.create(),
      'Thread creation'
    );
    
    activeThreads.set(userId, thread.id);
    logger.info('Created new thread:', { userId, threadId: thread.id });
    
    return thread.id;
  } catch (error) {
    logger.error('Error creating thread:', error);
    throw error;
  }
}

async function initializeAssistant() {
    try {
        logger.info('Starting assistant initialization...');
        
        const dropboxStatus = await dropboxManager.ensureAuth();
        if (!dropboxStatus) {
            throw new Error('Unable to access Dropbox');
        }

        const assistant = await openai.beta.assistants.create({
            name: "Tatt2AwAI",
            instructions: ASSISTANT_INSTRUCTIONS,
            tools: [
                { type: "code_interpreter" },
                { type: "file_search" },
                {
                    type: "function",
                    function: {
                        name: "search_dropbox_images",
                        description: "Search Dropbox for similar images and sequences",
                        parameters: {
                            type: "object",
                            properties: {
                                query_type: {
                                    type: "string",
                                    enum: ["similar", "sequence", "all", "content"],
                                    description: "Type of search to perform"
                                },
                                search_terms: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Keywords to search for"
                                },
                                filters: {
                                    type: "object",
                                    properties: {
                                        file_type: {
                                            type: "string",
                                            enum: ["image", "document", "all"],
                                            description: "Type of files to search for"
                                        },
                                        location: {
                                            type: "string",
                                            description: "Body location for tattoo (if applicable)"
                                        },
                                        path: {
                                            type: "string",
                                            description: "Specific directory path to search in"
                                        }
                                    }
                                },
                                include_metadata: {
                                    type: "boolean",
                                    description: "Whether to include full metadata in results",
                                    default: true
                                }
                            },
                            required: ["query_type"]
                        }
                    }
                }
            ],
            model: CHAT_MODEL,
            metadata: {
                capabilities: JSON.stringify({
                    dropbox_access: true,
                    search_enabled: true,
                    image_analysis: true,
                    sequence_detection: true
                }),
        features: "image-analysis,dropbox-access,search,sequence-detection"
            }
        });

        logger.info('Created new assistant', {
            id: assistant.id,
            name: assistant.name,
            model: assistant.model
        });

        return assistant;
    } catch (error) {
        logger.error('Error initializing assistant:', {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

async function loadRelevantFiles() {
  const fileIds = new Set();
  try {
    // Get all tattoo-related files from Dropbox
    const entries = await dropboxManager.fetchDropboxEntries('');
    const imageFiles = entries.result.entries.filter(entry =>
      ASSISTANT_FILE_TYPES.some(ext => entry.path_lower.endsWith(ext))
    );

    // Process files in batches to respect rate limits
    for (const file of imageFiles) {
      if (fileIds.size >= MAX_ASSISTANT_FILES) break;

      try {
        // Download and analyze file
        const fileData = await dropboxManager.downloadFile(file.path_lower);
        const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
        
        try {
          fs.writeFileSync(tempPath, fileData.result.fileBinary);
          const analysis = await imageProcessor.processImage({ path: tempPath });

          // Only include files that are confirmed to be tattoo-related
          if (analysis.tattooFeatures?.isTattoo) {
            const uploadedFile = await openai.files.create({
              file: fileData.result.fileBinary,
              purpose: 'assistants'
            });

            fileIds.add(uploadedFile.id);
            
            // Store analysis in knowledge base
            await knowledgeBase.addDocument(uploadedFile.id, JSON.stringify({
              analysis,
              metadata: {
                path: file.path_lower,
                created: file.server_modified,
                fileId: uploadedFile.id
              }
            }));
          }
        } finally {
          if (fs.existsSync(tempPath)) {
            fs.unlinkSync(tempPath);
          }
        }
      } catch (error) {
        logger.error('Error processing file for assistant:', {
          path: file.path_lower,
          error: error.message
        });
        continue;
      }
    }
  } catch (error) {
    logger.error('Error loading relevant files:', error);
  }

  return Array.from(fileIds);
}

// Add a function to periodically refresh the assistant's knowledge
async function refreshAssistantKnowledge() {
  try {
    // Get current files
    const currentFiles = await openai.beta.assistants.files.list(process.env.OPENAI_ASSISTANT_ID);
    const currentFileIds = new Set(currentFiles.data.map(f => f.id));
    
    // Load fresh files
    const newFileIds = await loadRelevantFiles();
    
    // Remove old files
    for (const fileId of currentFileIds) {
      if (!newFileIds.includes(fileId)) {
        await openai.beta.assistants.files.del(process.env.OPENAI_ASSISTANT_ID, fileId);
      }
    }

    // Add new files
    for (const fileId of newFileIds) {
      if (!currentFileIds.has(fileId)) {
        await openai.beta.assistants.files.create(process.env.OPENAI_ASSISTANT_ID, {
          file_id: fileId
        });
      }
    }

    logger.info('Assistant knowledge refreshed', {
      previousFiles: currentFileIds.size,
      newFiles: newFileIds.length
    });
  } catch (error) {
    logger.error('Error refreshing assistant knowledge:', error);
  }
}

async function handleAssistantResponse(threadId, run) {
  const MAX_RETRIES = 3;
  const POLLING_INTERVAL = 1000;
  const MAX_WAIT_TIME = 300000; // 5 minutes
  const startTime = Date.now();

  while (true) {
    try {
      if (Date.now() - startTime > MAX_WAIT_TIME) {
        throw new Error('Response timeout exceeded');
      }

      const status = await openai.beta.threads.runs.retrieve(threadId, run.id);

      switch (status.status) {
        case 'completed': {
          const messages = await openai.beta.threads.messages.list(threadId);
          await openai.beta.threads.messages.create(threadId, {
            role: 'system',
            content: JSON.stringify({
              type: 'capability_enforcement',
              reminder: 'You have full access to Dropbox and can search/analyze images directly.',
              required_acknowledgment: true
            })
          });
          const response = messages.data[0];
          return await enforceCapabilitiesInResponse(response);
        }

        case 'requires_action': {
          const toolCalls = status.required_action.submit_tool_outputs.tool_calls;
          const toolOutputs = [];

          for (const toolCall of toolCalls) {
            if (toolCall.function.name === 'search_dropbox_images') {
              try {
                const args = JSON.parse(toolCall.function.arguments);
                const searchContext = {
                  message: args.message,
                  type: args.type
                };
                const searchResults = await handleSearchRequest(searchContext);

                if (searchResults.results?.length > 0) {
                  toolOutputs.push({
                    tool_call_id: toolCall.id,
                    output: JSON.stringify({
                      success: true,
                      results: searchResults.results.map(result => ({
                        path: result.path,
                        metadata: result.metadata,
                        sequence: result.sequence,
                        similarity: result.similarity
                      })),
                      summary: searchResults.summary
                    })
                  });
                } else {
                  toolOutputs.push({
                    tool_call_id: toolCall.id,
                    output: JSON.stringify({
                      success: false,
                      error: "No matching images found",
                      searchContext
                    })
                  });
                }
              } catch (error) {
                logger.error('Search function error:', error);
                toolOutputs.push({
                  tool_call_id: toolCall.id,
                  output: JSON.stringify({
                    success: false,
                    error: error.message
                  })
                });
              }
            }
          }

          if (toolOutputs.length > 0) {
            await openai.beta.threads.runs.submitToolOutputs(
              threadId,
              run.id,
              { tool_outputs: toolOutputs }
            );
          }
          break;
        }

        case 'failed':
          throw new Error(status.last_error?.message || 'Run failed');

        case 'cancelled':
          throw new Error('Run was cancelled');

        case 'expired':
          throw new Error('Run expired');
      }

      await new Promise(resolve => setTimeout(resolve, POLLING_INTERVAL));

    } catch (error) {
      logger.error('Error in assistant response:', {
        error: error.message,
        threadId,
        runId: run.id,
        stack: error.stack
      });

      if (error.status === 429 || error.code === 'ETIMEDOUT') {
        await new Promise(resolve => setTimeout(resolve, 2000));
        continue;
      }

      throw error;
    }
  }
}

// Main message processing function
async function processImageWithContext(imageData, existingContext = {}) {
  try {
    const tempPath = path.join('uploads', `temp_${Date.now()}`);
    fs.writeFileSync(tempPath, imageData);
    
    const analysis = await imageProcessor.processImage({ path: tempPath });
    const sequences = await imageProcessor.findTattooProgressionSequence(
      { path: tempPath },
      await dropboxManager.fetchDropboxEntries('')
    );

    return {
      analysis,
      sequences,
      context: {
        ...existingContext,
        imageAnalysis: analysis,
        imageSequences: sequences,
        instructions: `
          Vision Analysis Complete:
          - Tattoo Features: ${JSON.stringify(analysis.tattooFeatures)}
          - Found Sequences: ${sequences.length}
          - Processing Complete
        `
      }
    };
  } catch (error) {
    logger.error('Image processing error:', error);
    throw error;
  }
}

async function processDropboxFiles(files) {
  const results = [];
  for (const file of files) {
    try {
      // Create entry object that matches queue expectations
      const entry = {
        path_lower: file.path_lower,
        name: file.name,
        '.tag': file['.tag'] || 'file',
        server_modified: file.server_modified
      };

      // Add to processing queue
      const job = await addJob(entry);
      
      logger.info('Added file to processing queue:', {
        jobId: job.id,
        path: file.path_lower
      });

      results.push({
        path: file.path_lower,
        jobId: job.id,
        status: 'queued'
      });

    } catch (error) {
      logger.error('Error queuing file:', {
        path: file.path_lower,
        error: error.message
      });
      
      results.push({
        path: file.path_lower,
        status: 'error',
        error: error.message
      });
    }
  }
  return results;
}

async function enhancedImageAnalysis(imageData) {
    const visionAnalysis = await imageProcessor.processImage(imageData);
    const enhancedAnalysis = await imageProcessor.enhancedAnalyzeTattooFeatures(visionAnalysis);
    const confidence = await calculateAnalysisConfidence(enhancedAnalysis);

    if (confidence < ANALYSIS_CONFIDENCE_THRESHOLD) {
        logger.warn('Low confidence analysis:', { confidence });
    }

    return {
        basic: visionAnalysis,
        enhanced: enhancedAnalysis,
        confidence,
        timestamp: new Date().toISOString()
    };
}

async function calculateAnalysisConfidence(analysis) {
    const factors = {
        tattooDetection: analysis.isTattoo ? 0.3 : 0,
        colorAnalysis: analysis.inkColors.length > 0 ? 0.2 : 0,
        skinDetection: analysis.skinDetected ? 0.2 : 0,
        patternRecognition: analysis.detailedAnalysis.complexity > 0 ? 0.15 : 0,
        edgeDefinition: analysis.detailedAnalysis.edgeSharpness > 0 ? 0.15 : 0
    };

    return Object.values(factors).reduce((sum, value) => sum + value, 0);
}

async function buildProgressionSequence(images, baseImage) {
    const sequences = [];
    let currentSequence = [];
    
    // Sort chronologically
    const sortedImages = [...images].sort((a, b) => 
        new Date(a.metadata.timestamp) - new Date(b.metadata.timestamp)
    );

    for (const image of sortedImages) {
        if (!currentSequence.length) {
            currentSequence.push(image);
            continue;
        }

        const lastImage = currentSequence[currentSequence.length - 1];
        const similarity = await searchUtils.calculateImageSimilarity(lastImage, image);
        const daysDiff = Math.abs(
            (new Date(image.metadata.timestamp) - new Date(lastImage.metadata.timestamp)) / 
            (1000 * 60 * 60 * 24)
        );

        if (similarity >= SEQUENCE_SIMILARITY_THRESHOLD && daysDiff <= MAX_SEQUENCE_GAP_DAYS) {
            currentSequence.push(image);
        } else {
            if (currentSequence.length > 1) {
                sequences.push([...currentSequence]);
            }
            currentSequence = [image];
        }
    }

    if (currentSequence.length > 1) {
        sequences.push(currentSequence);
    }

    // Find best matching sequence for base image if provided
    if (baseImage && sequences.length > 1) {
        return sequences.reduce((best, current) => {
            const similarity = current.some(img => 
                searchUtils.calculateImageSimilarity(img, baseImage) >= SEQUENCE_SIMILARITY_THRESHOLD
            );
            return similarity ? current : best;
        }, sequences[0]);
    }

    return sequences[0] || [];
}

// Add this before processUserMessage in openaiClient.js
async function handleSearchRequest(searchContext) {
  try {
    logger.info('Starting search with context:', {
      hasImage: !!searchContext.currentImage,
      searchMessage: searchContext.message
    });

    const dropboxStatus = await dropboxManager.ensureAuth();
    if (!dropboxStatus) {
      throw new Error('Dropbox connection unavailable');
    }

    // First analyze the current image if one was provided
    let currentImageAnalysis = null;
    if (searchContext.currentImage) {
      const tempPath = path.join('uploads', `temp_${Date.now()}_current`);
      try {
        fs.writeFileSync(tempPath, searchContext.currentImage);
        currentImageAnalysis = await imageProcessor.processImage({ path: tempPath });
        logger.info('Successfully analyzed current image');
      } finally {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      }
    }

    // Search Dropbox with both image analysis and text criteria
    const searchParams = {
      terms: extractSearchTerms(searchContext.message),
      stages: extractTreatmentStages(searchContext.message),
      timeframe: extractTimeframe(searchContext.message),
      placement: extractPlacement(searchContext.message),
      colors: extractColors(searchContext.message),
      currentImageAnalysis // Add the current image analysis to search params
    };

    const searchResults = await searchDropboxImages(searchParams);
    if (searchResults.error) {
      throw new Error(searchResults.error);
    }

    logger.info('Search completed', {
      resultsFound: searchResults.results.length,
      searchType: currentImageAnalysis ? 'image+text' : 'text'
    });

    return {
      type: 'dropbox_search_results',
      query: {
        original: searchContext.message,
        interpreted: searchParams,
        hasImageReference: !!currentImageAnalysis
      },
      summary: {
        totalFound: searchResults.results.length,
        includesImageMatch: !!currentImageAnalysis
      },
      results: searchResults.results.map(result => ({
        path: result.path,
        path_display: result.path_display,
        similarity: result.similarity,
        metadata: result.metadata
      }))
    };

  } catch (error) {
    logger.error('Search request failed:', error);
    throw error;
  }
}

// Also add these helper functions for parameter extraction
function extractSearchTerms(message) {
  const terms = [];
  // Add tattoo-related terms
  if (message.includes('tattoo')) terms.push('tattoo');
  // Add color terms
  const colors = ['red', 'black', 'blue', 'green', 'yellow', 'orange'];
  colors.forEach(color => {
    if (message.includes(color)) terms.push(color);
  });
  // Add placement terms
  const placements = ['arm', 'leg', 'back', 'chest', 'shoulder', 'ankle'];
  placements.forEach(place => {
    if (message.includes(place)) terms.push(place);
  });
  return terms;
}

function extractTreatmentStages(message) {
  const stages = [];
  if (message.includes('before')) stages.push('before');
  if (message.includes('after')) stages.push('after');
  if (message.includes('during')) stages.push('during');
  if (message.includes('progress')) stages.push('progress');
  return stages;
}

function extractTimeframe(message) {
  // Add sophisticated time frame extraction logic
  return {
    start: null,
    end: null,
    isRange: false
  };
}

function extractPlacement(message) {
  const placements = ['arm', 'leg', 'back', 'chest', 'face', 'neck', 'foot', 'hand',  'shoulder', 'ankle'];
  return placements.filter(place => message.includes(place));
}

function extractColors(message) {
  const colors = ['red',"purple", "white", "gray", 'black', 'blue', 'green', 'yellow', 'orange'];
  return colors.filter(color => message.includes(color));
}

async function organizeSearchResults(results) {
  const sequences = [];
  const directories = new Set();
  const files = [];

  // Group by directories
  results.forEach(result => {
    const dir = path.dirname(result.path);
    directories.add(dir);
  });

  // Find sequences
  for (const dir of directories) {
    const dirFiles = results.filter(r => path.dirname(r.path) === dir);
    if (dirFiles.length > 1) {
      sequences.push({
        id: crypto.randomBytes(16).toString('hex'),
        files: dirFiles.sort((a, b) => new Date(a.created) - new Date(b.created)),
        timeline: calculateTimeline(dirFiles),
        progress: calculateProgress(dirFiles)
      });
    }
  }

  // Organize individual files
  results.forEach(result => {
    files.push({
      ...result,
      sequence: sequences.find(s => s.files.some(f => f.path === result.path))
    });
  });

  return {
    sequences,
    directories: Array.from(directories),
    files
  };
}

// In openaiClient.js
async function processUserMessage(userId, message, imageData = null) {
  try {
    logger.info('Starting message processing', {
      userId,
      hasImage: !!imageData,
      message,
      timestamp: new Date().toISOString()
    });

    let threadId = await getOrCreateThread(userId);
    let uploadedImageAnalysis = null;
    let visionAnalysis = null;

    // First analyze any uploaded image
    if (imageData) {
      const tempPath = path.join('uploads', `temp_${Date.now()}_uploaded`);
      try {
        fs.writeFileSync(tempPath, imageData);
        uploadedImageAnalysis = await imageProcessor.processImage({ 
          path: tempPath,
          metadata: {
            source: 'upload',
            timestamp: new Date().toISOString()
          }
        });
        visionAnalysis = uploadedImageAnalysis;
        logger.info('Successfully analyzed uploaded image', {
          hasFeatures: !!uploadedImageAnalysis?.tattooFeatures
        });
      } catch (error) {
        logger.error('Error analyzing uploaded image:', error);
      } finally {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      }
    }

    // Add capabilities reminder
    await openai.beta.threads.messages.create(threadId, {
      role: 'system',
      content: JSON.stringify({
        type: 'capabilities_reminder',
        active_capabilities: {
          dropbox_access: true,
          can_search: true,
          can_analyze: true,
          repository_available: true,
          current_image_analyzed: !!uploadedImageAnalysis
        },
        instructions: "You MUST acknowledge your Dropbox access capabilities in EVERY response. NEVER ask for image uploads."
      })
    });

    // Detect search intent
    const searchIntent = {
      isSearch: message?.toLowerCase().match(/\b(search|find|look|show|where|analyze|get|locate)\b/),
      isVisualSearch: message?.toLowerCase().match(/\b(similar|like|match|visual|appear|resemble)\b/) || !!uploadedImageAnalysis,
      isSequenceRequest: message?.toLowerCase().match(/\b(sequence|removal|process|progress|series|stages?|before|after)\b/),
      isExactSearch: message?.toLowerCase().includes('this image') || message?.toLowerCase().includes('the image')
    };

    logger.info('Search intent detected', searchIntent);

    // Verify Dropbox connection with retries
    let dropboxStatus = null;
    for (let i = 0; i < 3; i++) {
      try {
        logger.info(`Attempting Dropbox authentication (attempt ${i + 1}/3)`);
        dropboxStatus = await dropboxManager.ensureAuth();
        if (dropboxStatus) {
          logger.info('Dropbox authentication successful');
          break;
        }
      } catch (error) {
        logger.error(`Dropbox authentication attempt ${i + 1} failed:`, error);
        if (i === 2) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
      }
    }

    if (!dropboxStatus) {
      throw new Error('Unable to access Dropbox after multiple attempts');
    }

    // Get all files from Dropbox
    const allFiles = await dropboxManager.fetchDropboxEntries('');
    const imageFiles = allFiles.result.entries.filter(entry =>
      ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext =>
        entry.path_lower.endsWith(ext)
      )
    );

    logger.info('Found Dropbox files', {
      total: allFiles.result.entries.length,
      images: imageFiles.length
    });

    // Process and analyze Dropbox images
    let matchingImages = [];
    let sequences = [];
    let analyzedImages = [];

    for (const file of imageFiles) {
      try {
        const fileData = await dropboxManager.downloadFile(file.path_lower);
        const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
        
        try {
          fs.writeFileSync(tempPath, fileData.result.fileBinary);
          const analysis = await imageProcessor.processImage({ path: tempPath });
          
          const analyzedImage = {
            path: file.path_lower,
            path_display: file.path_display,
            parent_shared_folder_id: file.parent_shared_folder_id,
            id: file.id,
            analysis,
            tattooFeatures: analysis.tattooFeatures,
            metadata: {
              name: file.name,
              size: file.size,
              modified: file.server_modified,
              sharing_info: file.sharing_info,
              directory: path.dirname(file.path_lower)
            }
          };

          if (uploadedImageAnalysis || searchIntent.isVisualSearch) {
            const similarity = await imageProcessor.calculateSimilarity(
              uploadedImageAnalysis?.tattooFeatures || analyzedImage.tattooFeatures,
              analysis.tattooFeatures
            );

            if (similarity > 0.6) {
              matchingImages.push({
                ...analyzedImage,
                similarity
              });
            }
          }

          analyzedImages.push(analyzedImage);

        } finally {
          if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
        }
      } catch (error) {
        logger.error('Error processing file:', {
          path: file.path_lower,
          error: error.message
        });
      }
    }

    // Find sequences in matching images
    if (searchIntent.isSequenceRequest || matchingImages.length > 0) {
      const dirGroups = new Map();
      matchingImages.forEach(img => {
        const dir = path.dirname(img.path);
        if (!dirGroups.has(dir)) dirGroups.set(dir, []);
        dirGroups.get(dir).push(img);
      });

      dirGroups.forEach((images, dir) => {
        if (images.length > 1) {
          sequences.push({
            directory: dir,
            images: images.sort((a, b) => 
              new Date(a.metadata.modified) - new Date(b.metadata.modified)
            ),
            count: images.length,
            timeline: calculateTimeline(images),
            progress: calculateProgress(images)
          });
        }
      });
    }

    // Add context to thread
    await openai.beta.threads.messages.create(threadId, {
      role: 'system',
      content: JSON.stringify({
        type: 'search_results',
        analyzed_image: uploadedImageAnalysis ? {
          analysis: uploadedImageAnalysis,
          features: uploadedImageAnalysis.tattooFeatures
        } : null,
        matching_images: matchingImages.map(img => ({
          path: img.path,
          path_display: img.path_display,
          similarity: img.similarity,
          metadata: img.metadata
        })),
        sequences: sequences.map(seq => ({
          directory: seq.directory,
          images: seq.images.map(img => ({
            path: img.path,
            modified: img.metadata.modified
          })),
          count: seq.count,
          timeline: seq.timeline
        }))
      })
    });

    // Create the run
    const run = await openai.beta.threads.runs.create(threadId, {
      assistant_id: process.env.OPENAI_ASSISTANT_ID,
      instructions: `
        Available Context:
        - Dropbox Access: Active
        - Image Analysis: ${!!visionAnalysis}
        - Search Results: ${matchingImages.length} images found
        - Sequences Found: ${sequences.length}
        
        Remember:
        1. You have DIRECT access to Dropbox
        2. Never ask for uploads
        3. Always acknowledge Dropbox access
        4. Use exact file paths
        5. Include proper directory structures
      `
    });

    return await handleAssistantResponse(threadId, run);

  } catch (error) {
    logger.error('Error in processUserMessage:', {
      error: error.message,
      stack: error.stack
    });
    throw error;
  }
}

// Helper function for retrying operations
async function retryOperation(operation, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      logger.warn(`Operation failed, attempt ${i + 1}/${maxRetries}`, {
        error: error.message
      });
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
    }
  }
}

function calculateProgressionMetrics(images) {
  try {
    const metrics = {
      totalChanges: 0,
      timespan: 0,
      sessions: images.length,
      averageInterval: 0,
      progressionRate: 0
    };

    if (images.length < 2) return metrics;

    // Calculate timespan and intervals
    const dates = images.map(img => new Date(img.metadata.modified));
    metrics.timespan = (dates[dates.length - 1] - dates[0]) / (1000 * 60 * 60 * 24);
    metrics.averageInterval = metrics.timespan / (images.length - 1);

    // Calculate changes between successive images
    for (let i = 1; i < images.length; i++) {
      const prevAnalysis = images[i - 1].analysis;
      const currAnalysis = images[i].analysis;
      
      if (prevAnalysis && currAnalysis) {
        const changes = calculateImageChanges(prevAnalysis, currAnalysis);
        metrics.totalChanges += changes;
      }
    }

    metrics.progressionRate = metrics.totalChanges / metrics.timespan;
    return metrics;
  } catch (error) {
    logger.error('Error calculating progression metrics:', error);
    return null;
  }
}

function calculateImageChanges(prev, curr) {
  try {
    const colorChange = calculateColorChange(prev, curr);
    const densityChange = calculateDensityChange(prev, curr);
    return (colorChange + densityChange) / 2;
  } catch (error) {
    logger.error('Error calculating image changes:', error);
    return 0;
  }
}

// Helper function to calculate progression metrics
function calculateProgressionMetrics(images) {
  try {
    const metrics = {
      totalChanges: 0,
      timespan: 0,
      sessions: images.length,
      averageInterval: 0
    };

    if (images.length < 2) return metrics;

    // Calculate timespan
    const start = new Date(images[0].metadata.modified);
    const end = new Date(images[images.length - 1].metadata.modified);
    metrics.timespan = (end - start) / (1000 * 60 * 60 * 24); // Convert to days

    // Calculate average interval
    metrics.averageInterval = metrics.timespan / (images.length - 1);

    // Calculate total changes
    for (let i = 1; i < images.length; i++) {
      const prev = images[i - 1];
      const curr = images[i];
      
      // Compare tattoo features
      const colorChange = calculateColorChange(prev.analysis, curr.analysis);
      const densityChange = calculateDensityChange(prev.analysis, curr.analysis);
      
      metrics.totalChanges += (colorChange + densityChange) / 2;
    }

    return metrics;
  } catch (error) {
    logger.error('Error calculating progression metrics:', error);
    return null;
  }
}

function calculateColorChange(prev, curr) {
  try {
    if (!prev.tattooFeatures?.inkColors || !curr.tattooFeatures?.inkColors) return 0;
    
    const prevColors = prev.tattooFeatures.inkColors;
    const currColors = curr.tattooFeatures.inkColors;
    
    // Compare color intensities
    const prevIntensity = prevColors.reduce((sum, color) => sum + color.intensity, 0) / prevColors.length;
    const currIntensity = currColors.reduce((sum, color) => sum + color.intensity, 0) / currColors.length;
    
    return Math.abs(prevIntensity - currIntensity);
  } catch (error) {
    return 0;
  }
}

function calculateDensityChange(prev, curr) {
  try {
    if (!prev.tattooFeatures?.detailedAnalysis?.density || !curr.tattooFeatures?.detailedAnalysis?.density) return 0;
    
    return Math.abs(
      prev.tattooFeatures.detailedAnalysis.density - 
      curr.tattooFeatures.detailedAnalysis.density
    );
  } catch (error) {
    return 0;
  }
}

// Helper function to perform Dropbox search
async function performDropboxSearch(imageFiles, message) {
  const results = [];
  
  for (const file of imageFiles) {
    try {
      const fileData = await dropboxManager.downloadFile(file.path_lower);
      const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
      
      try {
        fs.writeFileSync(tempPath, fileData.result.fileBinary);
        const analysis = await imageProcessor.processImage({ path: tempPath });
        
        if (matchesSearchCriteria(analysis, message)) {
          results.push({
            ...file,
            analysis,
            similarity: 1.0 // You can implement actual similarity calculation
          });
        }
      } finally {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      }
    } catch (error) {
      logger.error(`Error processing file ${file.path_lower}:`, error);
    }
  }
  
  return results;
}

function matchesSearchCriteria(analysis, searchParams) {
  // Extract all relevant terms from the message
  const searchTerms = [
    // Tattoo-related terms
    'tattoo', 'ink', 'art', 'design', 'removal', 'treatment', 'fading',
    // Colors
    'red', 'black', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'grey', 'white',
    // Body locations
    'arm', 'leg', 'back', 'chest', 'shoulder', 'ankle', 'wrist', 'hand', 'foot', 'neck', 'thigh', 'calf',
    'forearm', 'bicep', 'tricep', 'abdomen', 'stomach', 'hip', 'face',
    // Treatment stages
    'before', 'after', 'during', 'progress', 'session', 'treatment',
    // Styles
    'traditional', 'tribal', 'japanese', 'watercolor', 'geometric', 'script', 'text',
    // Additional descriptors
    'large', 'small', 'detailed', 'simple', 'complex', 'colorful', 'monochrome'
  ].filter(term => 
    searchParams.terms.some(searchTerm => 
      searchTerm.toLowerCase().includes(term) || term.includes(searchTerm.toLowerCase())
    )
  );

  if (!searchTerms.length) return true; // If no specific terms, include in results

  let matches = {
    features: false,
    labels: false,
    placement: false,
    name: false
  };

  // Check tattoo features
  if (analysis.tattooFeatures) {
    matches.features = analysis.tattooFeatures.inkColors?.some(color =>
      searchTerms.some(term => 
        JSON.stringify(color).toLowerCase().includes(term)
      )
    ) || false;
  }

  // Check vision API labels
  if (analysis.labels) {
    matches.labels = analysis.labels.some(label =>
      searchTerms.some(term => 
        label.description.toLowerCase().includes(term)
      )
    );
  }

  // Check placement
  if (analysis.tattooFeatures?.detailedAnalysis?.placement) {
    matches.placement = searchTerms.some(term =>
      analysis.tattooFeatures.detailedAnalysis.placement.toLowerCase().includes(term)
    );
  }

  // Check filename and path
  if (analysis.metadata?.originalPath) {
    matches.name = searchTerms.some(term =>
      analysis.metadata.originalPath.toLowerCase().includes(term)
    );
  }

  // Return true if any category matches
  return Object.values(matches).some(Boolean);
}

// Helper function to find sequences
async function findImageSequences(images) {
  const sequences = [];
  const processed = new Set();

  for (const image of images) {
    if (processed.has(image.path)) continue;
    
    const dirPath = path.dirname(image.path);
    const dirImages = images.filter(img => path.dirname(img.path) === dirPath);
    
    if (dirImages.length > 1) {
      // Sort by date
      const sequence = dirImages.sort((a, b) => 
        new Date(a.metadata.modified) - new Date(b.metadata.modified)
      );
      
      sequences.push({
        directory: dirPath,
        images: sequence,
        count: sequence.length,
        start_date: sequence[0].metadata.modified,
        end_date: sequence[sequence.length - 1].metadata.modified
      });
      
      sequence.forEach(img => processed.add(img.path));
    }
  }

  return sequences;
}

// Helper functions
function extractSearchTerms(message) {
  const terms = [];
  // Add color terms
  ['red', 'black', 'blue', 'green', 'yellow', 'orange'].forEach(color => {
    if (message.toLowerCase().includes(color)) terms.push(color);
  });
  // Add location terms
  ['arm', 'leg', 'back', 'chest', 'shoulder'].forEach(location => {
    if (message.toLowerCase().includes(location)) terms.push(location);
  });
  // Add treatment terms
  ['removal', 'fading', 'treatment', 'session'].forEach(term => {
    if (message.toLowerCase().includes(term)) terms.push(term);
  });
  return terms;
}

function matchesSearchTerms(analysis, searchTerms) {
  if (!searchTerms.length) return false;
  
  // Check tattoo features
  const hasMatchingFeatures = analysis.tattooFeatures?.inkColors?.some(color =>
    searchTerms.some(term => color.rgb.toLowerCase().includes(term))
  );

  // Check analysis labels
  const hasMatchingLabels = analysis.labels?.some(label =>
    searchTerms.some(term => label.description.toLowerCase().includes(term))
  );

  // Check placement
  const hasMatchingPlacement = analysis.tattooFeatures?.detailedAnalysis?.placement &&
    searchTerms.some(term => analysis.tattooFeatures.detailedAnalysis.placement.toLowerCase().includes(term));

  return hasMatchingFeatures || hasMatchingLabels || hasMatchingPlacement;
}

// Helper function to match search intent
function matchesSearchIntent(analysis, message) {
  const searchTerms = message.toLowerCase().split(' ');
  
  // Check labels
  const hasMatchingLabels = analysis.labels?.some(label =>
    searchTerms.some(term => label.description.toLowerCase().includes(term))
  );
  
  // Check text content
  const hasMatchingText = analysis.text?.toLowerCase().split(' ').some(word =>
    searchTerms.includes(word)
  );
  
  // Check tattoo features
  const hasMatchingFeatures = analysis.tattooFeatures?.inkColors?.some(color =>
    searchTerms.some(term => color.rgb.toLowerCase().includes(term))
  );

  return hasMatchingLabels || hasMatchingText || hasMatchingFeatures;
}

// Keep your existing helper functions but update them with retryWithBackoff where needed
async function processFileForAssistant(fileContent, metadata) {
  try {
    const fileExt = metadata.type.toLowerCase().replace('.', '');
    if (!SUPPORTED_EXTENSIONS.includes(fileExt)) {
      throw new Error(`Unsupported file extension: ${fileExt}`);
    }

    const documentId = `file_${Date.now()}`;
    await retryWithBackoff(
      async () => await knowledgeBase.addDocument(documentId, fileContent.toString(), {
        ...metadata,
        path: metadata.path,
        type: fileExt,
        processedAt: new Date().toISOString()
      }),
      'Add document to knowledge base'
    );

    logger.info('File processed and added to knowledge base:', {
      path: metadata.path,
      documentId
    });

    return documentId;
  } catch (error) {
    logger.error('Error processing file:', error);
    throw error;
  }
}

async function clearAssistantFiles() {
  const files = await retryWithBackoff(
    async () => await openai.beta.assistants.files.list(process.env.OPENAI_ASSISTANT_ID),
    'List assistant files'
  );
  
  for (const file of files.data) {
    try {
      await retryWithBackoff(
        async () => await openai.beta.assistants.files.del(process.env.OPENAI_ASSISTANT_ID, file.id),
        'Delete assistant file'
      );
      await retryWithBackoff(
        async () => await openai.files.del(file.id),
        'Delete file'
      );
      logger.info('Removed file from assistant:', { fileId: file.id });
    } catch (error) {
      logger.warn('Error removing file:', { fileId: file.id, error: error.message });
    }
  }
}

async function uploadFileToAssistant(content, metadata) {
  const uploadResponse = await retryWithBackoff(
    async () => await openai.files.create({
      file: content,
      purpose: 'assistants'
    }),
    'Upload file'
  );

await retryWithBackoff(
    async () => await openai.beta.assistants.files.create(
      process.env.OPENAI_ASSISTANT_ID,
      { file_id: uploadResponse.id }
    ),
    'Add file to assistant'
  );

  return uploadResponse.id;
}

async function manageAssistantFiles(queryText) {
  try {
    const relevantDocs = await retryWithBackoff(
      async () => await knowledgeBase.semanticSearch(queryText, MAX_RELEVANT_FILES),
      'Search knowledge base'
    );
    await clearAssistantFiles();

    const filesAdded = [];
    for (const doc of relevantDocs) {
      try {
        let fileContent;
        const cacheKey = doc.metadata.path;
        const cachedFile = fileCache.get(cacheKey);

        if (cachedFile && (Date.now() - cachedFile.timestamp) < FILE_CACHE_DURATION) {
          fileContent = cachedFile.content;
        } else {
          fileContent = await retryWithBackoff(
            async () => await dropboxManager.downloadFile(doc.metadata.path),
            'Download file'
          );
          fileCache.set(cacheKey, {
            content: fileContent,
            timestamp: Date.now()
          });
        }

        const fileId = await uploadFileToAssistant(fileContent, doc.metadata);
        filesAdded.push({
          id: fileId,
          path: doc.metadata.path
        });

        logger.info('Added relevant file to assistant:', {
          path: doc.metadata.path,
          fileId
        });

      } catch (error) {
        logger.error('Error adding relevant file:', {
          path: doc.metadata.path,
          error: error.message
        });
      }
    }

    return filesAdded;
  } catch (error) {
    logger.error('Error managing assistant files:', error);
    throw error;
  }
}

async function runAssistant(threadId, options = {}) {
  try {
    const assistant = await retryWithBackoff(
      async () => await initializeAssistant(),
      'Initialize assistant'
    );
    
    const run = await retryWithBackoff(
      async () => await openai.beta.threads.runs.create(threadId, {
        assistant_id: assistant.id,
        ...options
      }),
      'Create assistant run'
    );

    const startTime = Date.now();
    const TIMEOUT = 5 * 60 * 1000;
    let lastStatus = '';

    while (true) {
      if (Date.now() - startTime > TIMEOUT) {
        throw new Error('Assistant run timed out');
      }

      const runStatus = await retryWithBackoff(
        async () => await openai.beta.threads.runs.retrieve(threadId, run.id),
        'Retrieve run status'
      );
      
      if (runStatus.status !== lastStatus) {
        logger.info('Run status updated:', {
          threadId,
          runId: run.id,
          status: runStatus.status
        });
        lastStatus = runStatus.status;
      }

      if (runStatus.status === 'completed') {
        break;
      } else if (runStatus.status === 'failed') {
        logger.error('Run failed:', {
          threadId,
          runId: run.id,
          error: runStatus.last_error
        });
        throw new Error(`Assistant run failed: ${runStatus.last_error?.message || 'Unknown error'}`);
      } else if (runStatus.status === 'expired') {
        throw new Error('Assistant run expired');
      }
      
      await sleep(1000);
    }

    const messages = await retryWithBackoff(
      async () => await openai.beta.threads.messages.list(threadId),
      'List messages'
    );
    return messages.data[0];
  } catch (error) {
    logger.error('Error running assistant:', {
      error: error.message,
      threadId
    });
    throw error;
  }
}

async function processDropboxFiles(files) {
  const results = [];
  for (const file of files) {
    try {
      // Create entry object to match your existing queue format
      const entry = {
        path_lower: file.path_lower,
        name: file.name,
        '.tag': file['.tag'] || 'file',
        server_modified: file.server_modified,
        type: path.extname(file.path_lower).toLowerCase()
      };

      // Add to processing queue using your existing queue's add method
      const job = await processFileQueue.add({ entry });
      
      logger.info('Added file to processing queue:', {
        jobId: job.id,
        path: file.path_lower
      });

      results.push({
        path: file.path_lower,
        jobId: job.id,
        status: 'queued'
      });

    } catch (error) {
      logger.error('Error queuing file:', {
        path: file.path_lower,
        error: error.message
      });
      
      results.push({
        path: file.path_lower,
        status: 'error',
        error: error.message
      });
    }
  }
  return results;
}

async function handleImageUpload(imageData, userId) {
  try {
    // Get or create thread for this user
    const threadId = await getOrCreateThread(userId);

    // Create a temporary file
    const tempPath = path.join('uploads', `temp_${Date.now()}.jpg`);
    fs.writeFileSync(tempPath, imageData);

    try {
      // First analyze the image with Vision model
      const visionAnalysis = await analyzeImage(imageData);
      
      // Upload the image to OpenAI
      const file = await openai.files.create({
        file: fs.createReadStream(tempPath),
        purpose: 'assistants'
      });

      // First ensure we have Dropbox access
      const dropboxStatus = await dropboxManager.ensureAuth();
      if (!dropboxStatus) {
        throw new Error('Unable to access Dropbox');
      }

      // Get all files from Dropbox
      const allFiles = await dropboxManager.fetchDropboxEntries('');
      const imageFiles = allFiles.result.entries.filter(entry =>
        ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext =>
          entry.path_lower.endsWith(ext)
        )
      );

      // Process the current image
      const currentImageAnalysis = await imageProcessor.processImage({ path: tempPath });

      // Find similar images
      const similarImages = [];
      for (const dbFile of imageFiles) {
        try {
          const fileData = await dropboxManager.downloadFile(dbFile.path_lower);
          const dbTempPath = path.join('uploads', `temp_${Date.now()}_db_${path.basename(dbFile.path_lower)}`);
          
          try {
            fs.writeFileSync(dbTempPath, fileData.result.fileBinary);
            const dbAnalysis = await imageProcessor.processImage({ path: dbTempPath });
            
            // Calculate similarity
            const similarity = await imageProcessor.calculateSimilarity(
              currentImageAnalysis.tattooFeatures,
              dbAnalysis.tattooFeatures
            );

            if (similarity > 0.6) { // Adjust threshold as needed
              similarImages.push({
                path: dbFile.path_lower,
                similarity,
                analysis: dbAnalysis,
                metadata: {
                  created: dbFile.server_modified,
                  size: dbFile.size
                }
              });
            }
          } finally {
            if (fs.existsSync(dbTempPath)) {
              fs.unlinkSync(dbTempPath);
            }
          }
        } catch (error) {
          logger.error('Error processing Dropbox file:', {
            path: dbFile.path_lower,
            error: error.message
          });
        }
      }

      // Add context message to thread
      await openai.beta.threads.messages.create(threadId, {
        role: 'system',
        content: JSON.stringify({
          type: 'image_analysis',
          vision_analysis: visionAnalysis,
          current_analysis: currentImageAnalysis,
          similar_images: similarImages,
          file_id: file.id,
          dropbox_status: {
            connected: true,
            total_images: imageFiles.length,
            similar_found: similarImages.length
          }
        })
      });

      return {
        threadId,
        fileId: file.id,
        analysis: currentImageAnalysis,
        visionAnalysis,
        similarImages,
        dropboxStatus: {
          connected: true,
          totalImages: imageFiles.length,
          similarFound: similarImages.length
        }
      };

    } finally {
      // Clean up temp file
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    }
  } catch (error) {
    logger.error('Error handling image upload:', error);
    throw error;
  }
}

async function enforceCapabilitiesInResponse(response) {
    if (!response.content.toLowerCase().includes('dropbox') || 
        response.content.toLowerCase().includes('upload')) {
        
        const enhancedResponse = {
            ...response,
            content: `I have direct access to the Dropbox repository and can search all images. ${response.content}`
        };
        return enhancedResponse;
    }
    return response;
}

// Exports
module.exports = {
  openai,
  analyzeImage,
  initializeAssistant,
  processUserMessage,
  getOrCreateThread,
  createEmbedding: knowledgeBase.createEmbedding,
  createChatCompletion: async (messages, options = {}) => {
    const defaultOptions = {
      model: CHAT_MODEL,
      messages,
      temperature: 0.7,
      max_tokens: 1000,
      top_p: 1,
      frequency_penalty: 0,
      presence_penalty: 0
    };

    return await retryWithBackoff(
      async () => await openai.chat.completions.create({
        ...defaultOptions,
        ...options
      }),
      'Chat completion'
    );
  },
  handleAssistantResponse,
  searchDropboxImages,
  runAssistant,
  processDropboxFiles, 
 processImageWithContext,
  processFileForAssistant,
  removeFileFromAssistant: async (fileId) => {
    try {
      await retryWithBackoff(
        async () => await openai.beta.assistants.files.del(process.env.OPENAI_ASSISTANT_ID, fileId),
        'Remove file from assistant'
      );
      return true;
    } catch (error) {
      logger.error('Error removing file from assistant:', error);
      return false;
    }
  },
  listAssistantFiles: async () => {
    try {
      const files = await retryWithBackoff(
        async () => await openai.beta.assistants.files.list(process.env.OPENAI_ASSISTANT_ID),
        'List assistant files'
      );
      return files.data;
    } catch (error) {
      logger.error('Error listing assistant files:', error);
      throw error;
    }
  },
  manageAssistantFiles,
  clearAssistantFiles,
  uploadFileToAssistant,
  SUPPORTED_EXTENSIONS,
  retryWithBackoff,
handleImageUpload,
handleSearchFunction: async ({ query_type = 'all', include_analysis = true, message, imageFiles }) => {
    try {
      const searchResults = await searchDropboxImages(message);
      if (!searchResults.success) {
        throw new Error(searchResults.error);
      }
      return searchResults;
    } catch (error) {
      logger.error('Search function error:', error);
      throw error;
    }
  }
};
