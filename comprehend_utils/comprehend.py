"""
AWS Comprehend and Embedding Generation Utilities

This module provides functions for:
- Analyzing transcripts with AWS Comprehend (entities, key phrases, sentiment)
- Generating embeddings using Amazon Titan
- Storing results in the database

Dependencies: boto3, psycopg2
"""

import json
import logging
from typing import Dict, List, Any
import boto3
from .config import AWS_REGION, get_bedrock_runtime_client

# Setup logging
logger = logging.getLogger(__name__)


def analyze_transcript_with_comprehend(plain_text: str) -> Dict:
    """
    Analyze transcript with AWS Comprehend to extract entities and key phrases.
    
    AWS Comprehend has a 5000 byte limit for synchronous operations,
    so we'll analyze in chunks and aggregate results.
    
    Args:
        plain_text: Plain text transcript
        
    Returns:
        Dictionary with entities and key phrases
    """
    logger.info("Analyzing transcript with AWS Comprehend...")
    
    # Initialize Comprehend client
    comprehend_client = boto3.client('comprehend', region_name=AWS_REGION)
    
    # Analyze in 4500 char chunks (leave buffer for multi-byte characters)
    chunk_size = 4500
    all_entities = []
    all_key_phrases = []
    sentiment_scores = []
    
    # Process text in chunks
    num_chunks = (len(plain_text) + chunk_size - 1) // chunk_size
    logger.info(f"  Processing {num_chunks} chunks for Comprehend analysis...")
    
    for i in range(0, len(plain_text), chunk_size):
        chunk = plain_text[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        
        # Heartbeat logging for long operations
        if chunk_num % 5 == 0:
            logger.info(f"⏳ Processing Comprehend chunk {chunk_num}/{num_chunks}...")
        
        try:
            # Detect entities
            entities_response = comprehend_client.detect_entities(
                Text=chunk,
                LanguageCode='en'
            )
            all_entities.extend(entities_response['Entities'])
            
            # Extract key phrases
            key_phrases_response = comprehend_client.detect_key_phrases(
                Text=chunk,
                LanguageCode='en'
            )
            all_key_phrases.extend(key_phrases_response['KeyPhrases'])
            
            # Get sentiment for first chunk only (representative)
            if i == 0:
                sentiment_response = comprehend_client.detect_sentiment(
                    Text=chunk,
                    LanguageCode='en'
                )
                sentiment_scores.append({
                    'sentiment': sentiment_response['Sentiment'],
                    'scores': sentiment_response['SentimentScore']
                })
        
        except Exception as e:
            logger.warning(f"  Error analyzing chunk {i//chunk_size + 1}: {e}")
            continue
    
    # Deduplicate and rank entities by confidence
    entities_dict = {}
    for entity in all_entities:
        key = (entity['Text'].lower(), entity['Type'])
        if key not in entities_dict or entity['Score'] > entities_dict[key]['Score']:
            entities_dict[key] = entity
    
    ranked_entities = sorted(entities_dict.values(), key=lambda x: x['Score'], reverse=True)
    
    # Deduplicate and rank key phrases
    phrases_dict = {}
    for phrase in all_key_phrases:
        key = phrase['Text'].lower()
        if key not in phrases_dict or phrase['Score'] > phrases_dict[key]['Score']:
            phrases_dict[key] = phrase
    
    ranked_phrases = sorted(phrases_dict.values(), key=lambda x: x['Score'], reverse=True)
    
    logger.info(f"  Found {len(ranked_entities)} unique entities")
    logger.info(f"  Found {len(ranked_phrases)} unique key phrases")
    
    # Return top results
    return {
        'entities': ranked_entities[:100],  # Top 100 entities
        'key_phrases': ranked_phrases[:50],  # Top 50 key phrases
        'sentiment': sentiment_scores[0] if sentiment_scores else None
    }


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector using Amazon Titan Embeddings.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats (1024-dimensional vector)
    """
    bedrock_runtime = get_bedrock_runtime_client()
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({
            'inputText': text[:8000]  # Titan limit
        })
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']


def store_comprehend_analysis(conn, asset_id: str, comprehend_analysis: Dict) -> Dict:
    """
    Store AWS Comprehend analysis results in the database.
    
    Args:
        conn: Database connection
        asset_id: UUID of the asset
        comprehend_analysis: Dictionary with entities, key_phrases, and sentiment
        
    Returns:
        Dictionary with counts of stored items
    """
    cursor = conn.cursor()
    
    counts = {'entities': 0, 'key_phrases': 0, 'sentiment': 0}
    
    try:
        # 1. Store entities
        entities = comprehend_analysis.get('entities', [])
        if entities:
            # Delete existing entities for this asset
            cursor.execute("DELETE FROM comprehend_entities WHERE resource_id = %s", (asset_id,))
            
            # Insert new entities
            for entity in entities:
                cursor.execute("""
                    INSERT INTO comprehend_entities 
                    (resource_id, entity_text, entity_type, confidence_score, begin_offset, end_offset)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    asset_id,
                    entity['Text'],
                    entity['Type'],
                    entity['Score'],
                    entity.get('BeginOffset'),
                    entity.get('EndOffset')
                ))
                counts['entities'] += 1
        
        # 2. Store key phrases
        key_phrases = comprehend_analysis.get('key_phrases', [])
        if key_phrases:
            # Delete existing key phrases for this asset
            cursor.execute("DELETE FROM comprehend_key_phrases WHERE resource_id = %s", (asset_id,))
            
            # Insert new key phrases
            for phrase in key_phrases:
                cursor.execute("""
                    INSERT INTO comprehend_key_phrases 
                    (resource_id, phrase_text, confidence_score, begin_offset, end_offset)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    asset_id,
                    phrase['Text'],
                    phrase['Score'],
                    phrase.get('BeginOffset'),
                    phrase.get('EndOffset')
                ))
                counts['key_phrases'] += 1
        
        # 3. Store sentiment
        sentiment = comprehend_analysis.get('sentiment')
        if sentiment:
            # Delete existing sentiment for this asset
            cursor.execute("DELETE FROM comprehend_sentiment WHERE resource_id = %s", (asset_id,))
            
            # Insert new sentiment
            cursor.execute("""
                INSERT INTO comprehend_sentiment 
                (resource_id, sentiment, positive_score, negative_score, neutral_score, mixed_score)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                asset_id,
                sentiment['sentiment'],
                sentiment['scores']['Positive'],
                sentiment['scores']['Negative'],
                sentiment['scores']['Neutral'],
                sentiment['scores']['Mixed']
            ))
            counts['sentiment'] = 1
        
        conn.commit()
        logger.info(f"Stored Comprehend analysis: {counts['entities']} entities, {counts['key_phrases']} key phrases, {counts['sentiment']} sentiment")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing Comprehend analysis: {e}")
        raise
    
    finally:
        cursor.close()
    
    return counts


def store_embeddings(conn, asset_id: str, embeddings_data: List[Dict]) -> int:
    """
    Store embeddings in the database.
    
    Args:
        conn: Database connection
        asset_id: UUID of the asset
        embeddings_data: List of embedding dictionaries with keys:
            - content_type: Type of content (e.g., 'transcript', 'summary')
            - text: The text that was embedded
            - embedding: The vector embedding
            - metadata: Additional JSONB metadata
            
    Returns:
        Number of embeddings stored
    """
    cursor = conn.cursor()
    
    for data in embeddings_data:
        cursor.execute("""
            INSERT INTO asset_embeddings 
            (resource_id, content_type, embedding, embedding_model, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (resource_id, content_type, embedding_model)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """, (
            asset_id,
            data['content_type'],
            data['embedding'],
            'amazon.titan-embed-text-v2:0'
        ))
    
    conn.commit()
    logger.info(f"Stored {len(embeddings_data)} embeddings for asset {asset_id}")
    
    return len(embeddings_data)
