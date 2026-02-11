import os
import uuid
from datetime import datetime
from http import HTTPStatus

from bson.objectid import ObjectId
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from pymongo import MongoClient

from models import OrdinanceRetrievalSystem

# Initialize flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
bcrypt = Bcrypt(app)

# Configure JWT
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "MUjnHdptF4dX8c4dZ75X")
jwt = JWTManager(app)

uri = "mongodb+srv://admin:U3h9HJw6tRzurUqN@clusterjayomabot.twcwlf4.mongodb.net"

# MongoDB Atlas connection
client = MongoClient(os.getenv("MONGO_URI", uri))
db = client["test"]
users_collection = db["users"]
prompts_collection = db["prompts"]
categories_collection = db["categories"]

try:
    # Configuration for Phi-2
    LLM_MODEL_PATH = "zephyr-7b-beta.Q5_K_M.gguf"  # Update with your Phi-2 model path
    LLM_MODEL_TYPE = "zephyr"  # Using Phi-2 for faster inference

    print("🚀 Quick Start Mode - Loading Phi-2 RAG System")

    # Check if LLM model exists
    use_rag = os.path.exists(LLM_MODEL_PATH)

    if not use_rag:
        print(f"⚠️ LLM model not found at {LLM_MODEL_PATH}")
        print("💡 Download Zephyr GGUF model:")
        print("!wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf")
        print("📋 Falling back to classic retrieval mode...\n")
        # print(f"Zephyr GPU layers: {self.model.model.n_gpu_layers}")  # Should show 20

        # Initialize system without RAG
        retrieval_system = OrdinanceRetrievalSystem()
    else:
        print(f"✨ Loading RAG-enhanced system with {LLM_MODEL_TYPE} model...")
        # Initialize system with RAG
        retrieval_system = OrdinanceRetrievalSystem(
            llm_model_path=LLM_MODEL_PATH,
            llm_model_type=LLM_MODEL_TYPE
        )

    # Check if valid checkpoint exists
    checkpoint_path = './best_model'
    checkpoint_valid = os.path.exists(checkpoint_path)

    if checkpoint_valid:
        try:
            print("⏳ Loading existing optimized model...")
            retrieval_system.load_best_checkpoint()
            print("✅ Successfully loaded trained model")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {str(e)}")
            checkpoint_valid = False

    if not checkpoint_valid:
        print("\n🆕 Quick initialization (skipping optimization)")

        # Load data
        data_file = None
        try:
            # Check if running in Google Colab
            try:
                import google.colab

                is_colab = True
            except ImportError:
                is_colab = False

            # Load data based on environment
            if is_colab:
                try:
                    data_file = retrieval_system.upload_dataset_colab()
                except Exception as e:
                    print(f"Failed to upload file in Colab: {str(e)}")
                    exit(1)
            else:
                data_file = "ordinance_data.csv"
                if not os.path.exists(data_file):
                    print(f"Error: Data file '{data_file}' not found.")
                    print("Please ensure your CSV file is in the same directory as this script.")
                    exit(1)

            # Load the data
            print("\nLoading data...")
            df = retrieval_system.load_data(data_file)
            if df is None or len(df) == 0:
                raise ValueError("No data loaded or empty dataset")
            print(f"Successfully loaded {len(df)} ordinances")

        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            exit(1)

        # Initialize models
        try:
            print("\nInitializing models...")
            retrieval_system.initialize_models()
            print("Models initialized successfully")
        except Exception as e:
            print(f"Failed to initialize models: {str(e)}")
            exit(1)

        # Generate embeddings
        try:
            print("\nGenerating embeddings...")
            retrieval_system.generate_embeddings()
            print("Embeddings generated successfully")
        except Exception as e:
            print(f"Failed to generate embeddings: {str(e)}")
            exit(1)

        # SKIP OPTIMIZATION - Use default parameters
        print("\n⚡ Skipping optimization for faster setup")
        print("✅ System ready with default parameters!")

    # Run the enhanced chatbot interface
    # retrieval_system.interactive_chatbot_loop()
except Exception as e:
    print(f"Fatal error: {str(e)}")
    exit(1)


def category_to_dict(category):
    return {
        "id": str(category["_id"]),
        "name": category["name"]
    }


def user_to_dict(user):
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "role": user["role"]
    }


def conversation_to_dict(conversation):
    return {
        "id": str(conversation["_id"]),
        "chatId": conversation["chatId"],
        "user_id": conversation["user_id"],
        "messages": conversation["messages"],
        "status": conversation["status"]
    }


def process_command(user_input, conversation_history):
    """
    Mock AI model logic.
    Replace this logic with actual integration for your AI model.
    """
    return f"Bot response to: '{user_input}'. Based on conversation: {conversation_history}"


@app.route('/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        role = data.get("role")

        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already exists"}), HTTPStatus.BAD_REQUEST

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = {"name": name, "email": email, "password": hashed_password, "role": role}
        result = users_collection.insert_one(user)

        return jsonify(
            {"message": "User registered successfully", "userId": str(result.inserted_id)}), HTTPStatus.CREATED.value

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), HTTPStatus.BAD_REQUEST

        if not bcrypt.check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid password"}), HTTPStatus.BAD_REQUEST

        payload = {"id": str(user["_id"]), "name": user["name"], "email": user["email"], "role": user["role"]}
        token = create_access_token(identity=payload)

        return jsonify({"accessToken": token, "expiresIn": 3600}), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/users', methods=['GET'])
def get_all_users():
    try:
        users = users_collection.find()
        return jsonify([user_to_dict(user) for user in users]), HTTPStatus.OK
    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), HTTPStatus.NOT_FOUND

        return jsonify(user_to_dict(user)), HTTPStatus.OK
    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        data = request.json
        updated_data = {}

        if "name" in data:
            updated_data["name"] = data["name"]
        if "email" in data:
            if users_collection.find_one({"email": data["email"], "_id": {"$ne": ObjectId(user_id)}}):
                return jsonify({"error": "Email already in use"}), HTTPStatus.BAD_REQUEST

            updated_data["email"] = data["email"]
        if "password" in data:
            updated_data["password"] = bcrypt.generate_password_hash(data["password"]).decode('utf-8')
        if "role" in data:
            updated_data["role"] = data["role"]

        result = users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": updated_data})

        if result.matched_count == 0:
            return jsonify({"error": "User not found"}), HTTPStatus.NOT_FOUND

        return jsonify({"message": "User updated successfully"}), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), HTTPStatus.NOT_FOUND

        return jsonify({"message": "User deleted successfully"}), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


# @app.route('/prompts', methods=['POST'])
# def create_conversation():
#     try:
#         data = request.json
#         user_message = data.get("message", "")
#
#         if not user_message:
#             return jsonify({"error": "User message is required"}), 400
#
#         chat_id = str(uuid.uuid4())
#         timestamp = datetime.utcnow().isoformat()
#
#         # Bot's response: Use the retrieve_ordinances_with_details function
#         ai_response = retrieval_system.process_message(user_message)
#
#         # Handle errors from the ordinance system
#         if "error" in ai_response:
#             return jsonify({"error": ai_response["error"]}), 400
#
#         # Initialize the conversation
#         conversation = {
#             "chatId": chat_id,
#
#             "messages": [
#                 {"from": "user", "message": user_message, "timestamp": timestamp},
#                 {"from": "bot", "message": ai_response["message"], "results": ai_response["results"],
#                  "timestamp": timestamp}
#             ],
#             "status": "active"
#         }
#
#         # Insert into the database
#         prompts_collection.insert_one(conversation)
#
#         return jsonify(conversation_to_dict(conversation)), 201
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/prompts', methods=['POST'])
def create_conversation_with_user_id():
    try:
        data = request.json
        user_message = data.get("message", "")
        user_id = data.get("user_id", "")

        if not user_message:
            return jsonify({"error": "User message is required"}), 400

        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Bot's response: Use the retrieve_ordinances_with_details function
        ai_response = retrieval_system.process_message(user_message)

        # Handle errors from the ordinance system
        if "error" in ai_response:
            return jsonify({"error": ai_response["error"]}), 400



        # Initialize the conversation
        conversation = {
            "chatId": chat_id,
            "user_id": user_id,
            "messages": [
                {"from": "user", "message": user_message, "timestamp": timestamp},
                {"from": "bot", "message": ai_response["message"], "results": ai_response["results"],
                 "timestamp": timestamp}
            ],
            "status": "active"
        }

        # Insert into the database
        prompts_collection.insert_one(conversation)

        return jsonify(conversation_to_dict(conversation)), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prompts', methods=['GET'])
def get_all_conversations():
    try:
        prompts = prompts_collection.find(sort=[("_id", -1)])
        return jsonify([conversation_to_dict(prompt) for prompt in prompts]), HTTPStatus.OK
    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/prompts/<chat_id>', methods=['GET'])
def get_conversation_with_chat_id(chat_id):
    try:
        conversation = prompts_collection.find_one({"chatId": chat_id})
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify(conversation_to_dict(conversation))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prompts/user/<user_id>', methods=['GET'])
def get_conversations_by_user_id(user_id):
    try:
        # Method 1: Get ALL conversations for the user
        conversations = list(prompts_collection.find({"user_id": user_id}).sort("_id", -1))

        if not conversations:
            return jsonify({"error": "No conversations found for this user"}), 404

        # Convert all conversations to dict format
        conversations_list = [conversation_to_dict(conv) for conv in conversations]

        return jsonify(conversations_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prompts/user/<user_id>', methods=['DELETE'])
def delete_all_user_conversations(user_id):
    try:
        # First, check if the user has any conversations
        conversation_count = prompts_collection.count_documents({"user_id": user_id})

        if conversation_count == 0:
            return jsonify({"error": "No conversations found for this user"}), 404

        # Delete all conversations for the user
        result = prompts_collection.delete_many({"user_id": user_id})

        if result.deleted_count == 0:
            return jsonify({"error": "Failed to delete conversations"}), 500

        return jsonify({
            "message": f"Successfully deleted all conversations for user {user_id}",
            "deleted_count": result.deleted_count,
            "user_id": user_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prompts/<chat_id>', methods=['PUT'])
def update_conversation(chat_id):
    try:
        # Fetch the conversation by chatId
        conversation = prompts_collection.find_one({"chatId": chat_id})

        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Get the user's new message from the request body
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "User message is required"}), 400

        timestamp = datetime.utcnow().isoformat()

        # AI's response using retrieve_ordinances_with_details
        ai_response = retrieval_system.process_message(user_message)

        # Handle errors from the ordinance system
        if "error" in ai_response:
            return jsonify({"error": ai_response["error"]}), 400

        # Create new user and bot messages
        new_user_message = {"from": "user", "message": user_message, "timestamp": timestamp}
        new_bot_message = {"from": "bot", "message": ai_response["message"], "results": ai_response["results"],
                           "timestamp": timestamp}

        # Update the conversation in the database
        prompts_collection.update_one(
            {"chatId": chat_id},
            {"$push": {"messages": {"$each": [new_user_message, new_bot_message]}}}
        )

        # Return the updated conversation
        conversation["messages"].extend([new_user_message, new_bot_message])

        return jsonify(conversation_to_dict(conversation)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prompts/<chat_id>', methods=['DELETE'])
def delete_conversation(chat_id):
    try:
        result = prompts_collection.delete_one({"chatId": chat_id})
        if result.deleted_count == 0:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify({"message": "Conversation deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Create a new category
@app.route('/categories', methods=['POST'])
def create_category():
    try:
        data = request.json
        name = data.get("name")

        if not name:
            return jsonify({"error": "Category name is required"}), HTTPStatus.BAD_REQUEST

        # Check if category with same name exists
        if categories_collection.find_one({"name": name}):
            return jsonify({"error": "Category with this name already exists"}), HTTPStatus.BAD_REQUEST

        category = {"name": name}
        result = categories_collection.insert_one(category)

        created_category = categories_collection.find_one({"_id": result.inserted_id})
        return jsonify(category_to_dict(created_category)), HTTPStatus.CREATED

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


# Get all categories
@app.route('/categories', methods=['GET'])
def get_all_categories():
    try:
        categories = categories_collection.find()
        return jsonify([category_to_dict(category) for category in categories]), HTTPStatus.OK
    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


# Get a single category by ID
@app.route('/categories/<category_id>', methods=['GET'])
def get_category(category_id):
    try:
        category = categories_collection.find_one({"_id": ObjectId(category_id)})
        if not category:
            return jsonify({"error": "Category not found"}), HTTPStatus.NOT_FOUND

        return jsonify(category_to_dict(category)), HTTPStatus.OK
    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


# Update a category
@app.route('/categories/<category_id>', methods=['PUT'])
def update_category(category_id):
    try:
        data = request.json
        name = data.get("name")

        if not name:
            return jsonify({"error": "Category name is required"}), HTTPStatus.BAD_REQUEST

        # Check if another category with the same name exists
        existing = categories_collection.find_one({
            "name": name,
            "_id": {"$ne": ObjectId(category_id)}
        })
        if existing:
            return jsonify({"error": "Category with this name already exists"}), HTTPStatus.BAD_REQUEST

        result = categories_collection.update_one(
            {"_id": ObjectId(category_id)},
            {"$set": {"name": name}}
        )

        if result.matched_count == 0:
            return jsonify({"error": "Category not found"}), HTTPStatus.NOT_FOUND

        updated_category = categories_collection.find_one({"_id": ObjectId(category_id)})
        return jsonify(category_to_dict(updated_category)), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


# Delete a category
@app.route('/categories/<category_id>', methods=['DELETE'])
def delete_category(category_id):
    try:
        result = categories_collection.delete_one({"_id": ObjectId(category_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Category not found"}), HTTPStatus.NOT_FOUND

        return jsonify({"message": "Category deleted successfully"}), HTTPStatus.OK
    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == '__main__':
    app.run(debug=True)
