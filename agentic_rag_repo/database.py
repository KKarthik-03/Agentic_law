# database.py --> MongoDB database.py

import datetime
import hashlib
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from config import MONGO_URI

@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        db = client["Agentic_RAG"]
        client.admin.command('ping')
        return db
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(db, username: str, email: str, password: str) -> bool:
    """Create new user"""
    try:
        users_collection = db["users"]
        
        # Check if user already exists
        if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            return False
        
        user_doc = {
            "username": username,
            "email": email,
            "password": hash_password(password),
            "created_at": datetime.datetime.now(),
            "last_login": datetime.datetime.now()
        }
        
        users_collection.insert_one(user_doc)
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False

def authenticate_user(db, username: str, password: str) -> dict:
    """Authenticate user"""
    try:
        users_collection = db["users"]
        user = users_collection.find_one({"username": username})
        
        if user and user["password"] == hash_password(password):
            # Update last login
            users_collection.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.datetime.now()}}
            )
            return user
        return None
    except Exception as e:
        st.error(f"Error authenticating user: {e}")
        return None

def save_chat_message(db, user_id, chat_id, message_type, content, metadata=None):
    """Save chat message to MongoDB"""
    try:
        chats_collection = db["chats"]
        
        message = {
            "user_id": user_id,
            "chat_id": chat_id,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.datetime.now(),
            "metadata": metadata or {}
        }
        
        chats_collection.insert_one(message)
    except Exception as e:
        st.error(f"Error saving message: {e}")

def get_user_chats(db, user_id):
    """Get all chats for a user"""
    try:
        chats_collection = db["chats"]
        
        # Get unique chat_ids for the user with latest message
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$chat_id",
                "latest_message": {"$first": "$content"},
                "latest_timestamp": {"$first": "$timestamp"}
            }},
            {"$sort": {"latest_timestamp": -1}}
        ]
        
        return list(chats_collection.aggregate(pipeline))
    except Exception as e:
        st.error(f"Error getting chats: {e}")
        return []

def get_chat_messages(db, chat_id):
    """Get all messages for a specific chat"""
    try:
        chats_collection = db["chats"]
        return list(chats_collection.find({"chat_id": chat_id}).sort("timestamp", 1))
    except Exception as e:
        st.error(f"Error getting chat messages: {e}")
        return []