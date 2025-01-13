const { createClient } = require('@supabase/supabase-js');
const logger = require('./logger');
const { createEmbedding } = require('./services');
const crypto = require('crypto');
const path = require('path');

require('dotenv').config();

// Constants for processing
const CHUNK_SIZE = 4000;
const CHUNK_OVERLAP = 500;
const MAX_BATCH_SIZE = 3;
const MAX_RETRIES = 3;
const SIMILARITY_THRESHOLD = 0.75;

// Initialize Supabase client
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

class KnowledgeBase {
  constructor() {
    this.processingQueue = [];
    this.isProcessing = false;
    this.imageCache = new Map();
  }

  async validateConnection() {
    try {
      const { data, error } = await supabase.from('documents').select('count').limit(1);
      if (error) throw error;
      return true;
    } catch (error) {
      logger.error('Supabase connection error:', error);
      return false;
    }
  }

  async extractKnowledge(analysisResult, fileName) {
    try {
      logger.info('Extracting knowledge from tattoo analysis', { fileName });

      if (!analysisResult) {
        throw new Error('No analysis results provided');
      }

      // Extract key tattoo-specific information
      const content = {
        tattooFeatures: analysisResult.tattooFeatures || {},
        colorProfile: analysisResult.colors || [],
        labels: analysisResult.labels || [],
        text: analysisResult.text || '',
        objects: analysisResult.objects || []
      };

      const metadata = {
        fileName,
        type: 'tattoo-analysis',
        processingDate: new Date().toISOString(),
        tattooFeatures: analysisResult.tattooFeatures || {},
        progressionMetrics: analysisResult.progressionMetrics || {},
        hasAnalysis: true
      };

      // Generate unique ID for the document
      const id = `tattoo_${fileName}_${Date.now()}`;

      // Store in knowledge base
      await this.addDocument(id, JSON.stringify(content), metadata);

      // If this is part of a sequence, update sequence information
      if (analysisResult.isPartOfSequence) {
        await this.updateSequenceInformation(id, analysisResult.sequenceInfo);
      }

      return {
        id,
        content,
        metadata
      };
    } catch (error) {
      logger.error('Error extracting knowledge:', {
        error: error.message,
        fileName,
        stack: error.stack
      });
      throw error;
    }
  }

async addDocument(id, content, metadata = {}) {
    try {
        // Ensure metadata is properly structured and includes fileName
        metadata = {
            ...metadata,
            fileName: metadata.fileName || (metadata.path ? path.basename(metadata.path) : `doc_${id}`),
            processedAt: metadata.processedAt || new Date().toISOString()
        };

        // Log document info (truncated for readability)
        console.log('Adding document to knowledge base:', {
            id,
            content: content.substring(0, 100) + '...', // Log first 100 chars only
            metadata: JSON.stringify(metadata, null, 2)
        });

        // Log basic info to structured logger
        logger.info('Adding document to knowledge base:', { id, metadata });
        
        // Generate text chunks
        const chunks = await this.chunkText(content, metadata);
        if (chunks.length === 0) {
            logger.warn('No valid chunks generated from document:', { id });
            return { id, chunkCount: 0 };
        }

        // Process chunks in batches
        for (let i = 0; i < chunks.length; i += MAX_BATCH_SIZE) {
            const batch = chunks.slice(i, i + MAX_BATCH_SIZE);
            await Promise.all(batch.map(chunk => this.processChunk(chunk, id)));

            // Log progress for large documents
            if (chunks.length > MAX_BATCH_SIZE) {
                logger.info('Batch processing progress:', {
                    id,
                    processed: Math.min((i + MAX_BATCH_SIZE), chunks.length),
                    total: chunks.length
                });
            }
        }

        // Return success result
        return {
            id,
            chunkCount: chunks.length,
            metadata: {
                fileName: metadata.fileName,
                processedAt: metadata.processedAt,
                totalChunks: chunks.length
            }
        };

    } catch (error) {
        logger.error(`Error adding document ${id}:`, {
            error: error.message,
            stack: error.stack,
            metadata: metadata
        });
        throw error;
    }
}

  async updateSequenceInformation(imageId, sequenceInfo) {
    try {
      await supabase.from('image_sequences').upsert({
        image_id: imageId,
        sequence_group: sequenceInfo.groupId,
        sequence_order: sequenceInfo.position,
        metadata: sequenceInfo
      });
    } catch (error) {
      logger.error('Error updating sequence information:', error);
    }
  }

  async chunkText(text, metadata) {
if (!metadata || typeof metadata !== 'object') {
    metadata = {};
  }
  
  const fileName = metadata.fileName || 'unnamed_document';  
  const chunks = [];
    let startIndex = 0;
    
    while (startIndex < text.length) {
      let endIndex = startIndex + CHUNK_SIZE;
      if (endIndex < text.length) {
        const searchText = text.substring(startIndex, endIndex + 100);
        const match = searchText.match(/[.!?]\s+/);
        if (match) {
          endIndex = startIndex + match.index + 1;
        }
      } else {
        endIndex = text.length;
      }
      
      const chunk = text.substring(startIndex, endIndex).trim();
      if (chunk) {
        const chunkId = crypto.createHash('md5')
          .update(`${metadata.fileName}_${startIndex}_${Date.now()}`)
          .digest('hex');
        
        chunks.push({
          id: chunkId,
          content: chunk,
          metadata: {
            ...metadata,
            chunkIndex: chunks.length,
            startIndex,
            endIndex
          }
        });
      }
      
      startIndex = endIndex - CHUNK_OVERLAP;
    }
    
    return chunks;
  }

  async processChunk(chunk, parentId) {
    let retries = 0;
    while (retries < MAX_RETRIES) {
      try {
        const embedding = await createEmbedding(chunk.content);
        
        await supabase.from('documents').upsert({
          id: chunk.id,
          content: chunk.content,
          metadata: chunk.metadata,
          embedding,
          parent_id: parentId,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
        
        return true;
      } catch (error) {
        retries++;
        logger.error(`Error processing chunk (attempt ${retries}):`, {
          error: error.message,
          chunkId: chunk.id
        });
        
        if (retries === MAX_RETRIES) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retries)));
      }
    }
  }

  async searchSimilarImages(query, options = {}) {
    try {
      const { limit = 5, includeMetadata = true } = options;
      
      const queryEmbedding = await createEmbedding(query);
      const { data: matches } = await supabase.rpc('match_documents', {
        query_embedding: queryEmbedding,
        match_threshold: SIMILARITY_THRESHOLD,
        match_count: limit * 2
      });

      const results = matches
        .filter(match => match.metadata?.type === 'tattoo-analysis')
        .map(match => ({
          id: match.id,
          similarity: match.similarity,
          metadata: includeMetadata ? match.metadata : undefined,
          content: JSON.parse(match.content)
        }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);

      return results;
    } catch (error) {
      logger.error('Error searching similar images:', error);
      return [];
    }
  }

  async findProgressionSequence(imageId) {
    try {
      const { data: sequence } = await supabase
        .from('image_sequences')
        .select('*')
        .eq('image_id', imageId)
        .single();

      if (sequence) {
        const { data: relatedImages } = await supabase
          .from('image_sequences')
          .select('*')
          .eq('sequence_group', sequence.sequence_group)
          .order('sequence_order');

        return relatedImages || [];
      }

      return [];
    } catch (error) {
      logger.error('Error finding progression sequence:', error);
      return [];
    }
  }
}

module.exports = new KnowledgeBase();
