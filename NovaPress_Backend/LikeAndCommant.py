# app.py
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
# Enable CORS for all origins during development. For production, specify your frontend URL.
CORS(app) 

# MongoDB connection details
# It's highly recommended to use environment variables for sensitive info
MONGO_URI = os.getenv("MONGODB_URL", "mongodb+srv://jewelat50:jewel1234@cluster0.hvluq.mongodb.net/")
DB_NAME = os.getenv("MONGODB_DBNAME", "novapress_db")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# MongoDB Collections
articles_collection = db.articles
likes_collection = db.likes
comments_collection = db.comments

@app.route('/')
def home():
    """Basic route to confirm API is running."""
    return "News Interaction API is running!"

@app.route('/articles/id/<article_id>', methods=['GET'])
def get_article_by_id(article_id):
    """
    Fetches a single article by its MongoDB ObjectId.
    NOTE: While this endpoint is here, your frontend's main article fetch
    is configured to use VITE_API_BASE_URL (http://localhost:8000).
    This Flask endpoint will only be used if you change the frontend configuration.
    """
    try:
        if not ObjectId.is_valid(article_id):
            return jsonify({"error": "Invalid Article ID format"}), 400

        article = articles_collection.find_one({"_id": ObjectId(article_id)})
        if article:
            article['_id'] = str(article['_id']) # Convert ObjectId to string for JSON serialization
            # Ensure likes_count and comments_count are present, defaulting to 0
            article['likes_count'] = article.get('likes_count', 0)
            article['comments_count'] = article.get('comments_count', 0)
            # Handle datetime objects for JSON serialization
            for key in ["scraped_at", "published_at"]:
                if key in article and isinstance(article[key], datetime):
                    article[key] = article[key].isoformat()
            return jsonify(article), 200
        else:
            return jsonify({"message": "Article not found"}), 404
    except Exception as e:
        app.logger.error(f"Error fetching article {article_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/articles/<article_id>/like', methods=['POST'])
def toggle_like(article_id):
    """Allows a user to like or unlike an article."""
    user_id = request.json.get('user_id') # Assuming user_id is sent from frontend

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    if not ObjectId.is_valid(article_id):
        return jsonify({"error": "Invalid Article ID format"}), 400

    try:
        like_query = {"article_id": ObjectId(article_id), "user_id": user_id}
        existing_like = likes_collection.find_one(like_query)

        if existing_like:
            # User already liked, so unlike it
            likes_collection.delete_one(like_query)
            articles_collection.update_one(
                {"_id": ObjectId(article_id)},
                {"$inc": {"likes_count": -1}}
            )
            app.logger.info(f"User {user_id} unliked article {article_id}")
            return jsonify({"message": "Article unliked", "isLiked": False}), 200
        else:
            # User has not liked, so like it
            likes_collection.insert_one({
                "article_id": ObjectId(article_id),
                "user_id": user_id,
                "timestamp": datetime.now()
            })
            articles_collection.update_one(
                {"_id": ObjectId(article_id)},
                {"$inc": {"likes_count": 1}}
            )
            app.logger.info(f"User {user_id} liked article {article_id}")
            return jsonify({"message": "Article liked", "isLiked": True}), 201
    except Exception as e:
        app.logger.error(f"Error toggling like for article {article_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/articles/<article_id>/comments', methods=['POST'])
def add_comment(article_id):
    """Adds a new comment to an article."""
    author = request.json.get('author', 'Anonymous')
    text = request.json.get('text')

    if not text:
        return jsonify({"error": "Comment text is required"}), 400
    if not ObjectId.is_valid(article_id):
        return jsonify({"error": "Invalid Article ID format"}), 400

    try:
        new_comment = {
            "article_id": ObjectId(article_id),
            "author": author,
            "text": text,
            "timestamp": datetime.now()
        }
        comments_collection.insert_one(new_comment)
        
        # Increment comments_count in the articles_collection
        articles_collection.update_one(
            {"_id": ObjectId(article_id)},
            {"$inc": {"comments_count": 1}}
        )

        app.logger.info(f"Comment added to article {article_id} by {author}")
        return jsonify({
            "message": "Comment added successfully", 
            "comment": {
                "id": str(new_comment['_id']),
                "article_id": str(new_comment['article_id']),
                "author": new_comment['author'],
                "text": new_comment['text'],
                "timestamp": new_comment['timestamp'].isoformat() # Return ISO format for consistency
            }
        }), 201
    except Exception as e:
        app.logger.error(f"Error adding comment to article {article_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/articles/<article_id>/comments', methods=['GET'])
def get_comments(article_id):
    """Fetches all comments for a given article."""
    try:
        if not ObjectId.is_valid(article_id):
            return jsonify({"error": "Invalid Article ID format"}), 400

        comments_cursor = comments_collection.find({"article_id": ObjectId(article_id)}).sort("timestamp", -1)
        comments_list = []
        for comment in comments_cursor:
            comment['_id'] = str(comment['_id'])
            comment['article_id'] = str(comment['article_id'])
            # Convert datetime object to string for JSON serialization
            comment['timestamp'] = comment['timestamp'].isoformat() if isinstance(comment['timestamp'], datetime) else comment['timestamp']
            comments_list.append(comment)
        return jsonify(comments_list), 200
    except Exception as e:
        app.logger.error(f"Error fetching comments for article {article_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/articles/<article_id>/like_status', methods=['GET'])
def get_like_status(article_id):
    """Checks if a specific user has liked a given article."""
    user_id = request.args.get('user_id') # Get user_id from query parameter
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    if not ObjectId.is_valid(article_id):
        return jsonify({"error": "Invalid Article ID format"}), 400

    try:
        like_exists = likes_collection.find_one({
            "article_id": ObjectId(article_id),
            "user_id": user_id
        })
        return jsonify({"article_id": article_id, "user_id": user_id, "isLiked": bool(like_exists)}), 200
    except Exception as e:
        app.logger.error(f"Error checking like status for article {article_id} and user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Default host/port for development. Use '0.0.0.0' to make it accessible externally
    app.run(host='127.0.0.1', port=5000, debug=True)