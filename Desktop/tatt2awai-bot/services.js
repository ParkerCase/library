const { OpenAI } = require('openai');
const logger = require('./logger');

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Shared embedding function
async function createEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text
    });
    return response.data[0].embedding;
  } catch (error) {
    logger.error('Error creating embedding:', error);
    throw error;
  }
}

module.exports = {
  openai,
  createEmbedding
};
