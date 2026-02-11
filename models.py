import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
import torch
import json
import logging
import optuna
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer, CrossEncoder, losses, InputExample
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import time
import re
import random
from collections import Counter
from llama_cpp import Llama
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ordinance_retrieval.log')
    ]
)
logger = logging.getLogger("OrdinanceRetrieval")

class RAGOrdinanceLLM:
    """Handles LLM-based RAG responses for ordinance queries with performance optimizations"""

    def __init__(self, model_path: str = None, model_type: str = "zephyr"):
        """
        Initialize the local LLM for RAG with optimized settings

        Args:
            model_path: Path to the GGUF model file
            model_type: Type of model ("mistral", "phi", "llama", "zephyr")
        """
        self.model_type = model_type
        self.model = None
        self.context_window = 4096  # Zephyr supports larger context
        self.max_new_tokens = 200  # Slightly more tokens for better responses

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Optimized model-specific parameters for speed
        self.model_configs = {
            "mistral": {
                "n_ctx": 1024,
                "n_threads": 4,
                "n_gpu_layers": 20,
                "n_batch": 128,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20,
                "repeat_penalty": 1.1,
                "use_mmap": True,
                "use_mlock": False
            },
            "phi": {
                "n_ctx": 1024,
                "n_threads": 4,
                "n_gpu_layers": 32,
                "n_batch": 128,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20,
                "repeat_penalty": 1.1,
                "use_mmap": True,
                "use_mlock": False
            },
            "llama": {
                "n_ctx": 1024,
                "n_threads": 4,
                "n_gpu_layers": 20,
                "n_batch": 128,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20,
                "repeat_penalty": 1.1,
                "use_mmap": True,
                "use_mlock": False
            },
            "zephyr": {
                "n_ctx": 2048,  # Zephyr supports this
                "n_threads": 4,  # Reduce if on CPU
                "n_gpu_layers": -1,  # Adjusted based on VRAM
                "n_batch": 128,  # Reduced from 512
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "mirostat_mode": 2,  # Enable for better quality
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "use_mmap": True,
                "use_mlock": False
            }
        }

        # System prompts for different scenarios
        self.system_prompts = {
            "general": "You are Jayoma Bot. Answer briefly about Manila City ordinances.",
            "fines": "You are Jayoma Bot. State fines concisely.",
            "legal": "You are Jayoma Bot. Give brief legal info.",
            "process": "You are Jayoma Bot. List steps clearly."
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the GGUF model with optimized settings"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            config = self.model_configs.get(self.model_type, self.model_configs["zephyr"])

            logger.info(f"Loading {self.model_type} model from {model_path}...")
            self.model = Llama(
                model_path=model_path,
                n_ctx=config["n_ctx"],
                n_threads=config["n_threads"],
                n_gpu_layers=config["n_gpu_layers"],
                n_batch=config.get("n_batch", 512),
                verbose=False,
                use_mmap=config.get("use_mmap", True),
                use_mlock=config.get("use_mlock", False),
                rope_freq_scale=1.0,
                rope_freq_base=10000.0
            )
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def create_rag_prompt(self, query: str, context: List[Dict],
                         intent: Dict, conversation_history: List[Dict] = None) -> str:
        """
        Create an optimized prompt for the LLM based on query and retrieved ordinances
        """
        # Select appropriate system prompt based on intent
        if intent['type'] == 'fines':
            system_prompt = self.system_prompts["fines"]
        elif intent['type'] in ['details', 'explain']:
            system_prompt = self.system_prompts["legal"]
        elif 'process' in query.lower() or 'how to' in query.lower():
            system_prompt = self.system_prompts["process"]
        else:
            system_prompt = self.system_prompts["general"]

        # Format context documents - Use top 2 for Zephyr (better quality)
        context_text = ""
        if context:
            for i, doc in enumerate(context[:2]):  # Use top 2 documents
                context_text += f"\nOrdinance #{doc['ordinance_id']}: {doc['short_text'][:250]}"
                if doc.get('fines') and doc['fines'] != "N/A":
                    context_text += f"\nFines: {doc['fines'][:150]}"
                if i < len(context) - 1:
                    context_text += "\n"

        # Create the prompt based on model type
        if self.model_type == "zephyr":
            # Zephyr-7B-beta prompt format
            prompt = f"""<|system|>
{system_prompt}

Context Information:
{context_text}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

        elif self.model_type == "phi":
            # Phi-2 prompt format
            prompt = f"""Instruct: {system_prompt}

Context: {context_text}

User: {query}
Output:"""

        elif self.model_type == "mistral":
            # Mistral format
            prompt = f"""[INST] {system_prompt}
{context_text}

User Question: {query}

Give a brief, helpful response. [/INST]"""

        else:  # llama or default
            prompt = f"""### System: {system_prompt}
{context_text}

### Human: {query}

### Assistant: I'll provide a brief response.

"""

        return prompt

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the LLM with adjusted token limits"""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            config = self.model_configs.get(self.model_type, self.model_configs["zephyr"])

            # Adjusted generation parameters for better completion
            generation_params = {
                "max_tokens": kwargs.get("max_tokens", 512),  # Increased from 200
                "temperature": kwargs.get("temperature", config["temperature"]),
                "top_p": kwargs.get("top_p", config["top_p"]),
                "top_k": kwargs.get("top_k", config["top_k"]),
                "repeat_penalty": kwargs.get("repeat_penalty", config["repeat_penalty"]),
                "stop": kwargs.get("stop", ["</s>", "<|user|>", "<|system|>", "\n\nHuman:", "\n\nUser:", "###"]),
                "echo": False,
                "stream": False
            }

            # Add Zephyr-specific parameters if using Zephyr
            if self.model_type == "zephyr" and "mirostat_mode" in config:
                generation_params.update({
                    "mirostat_mode": config["mirostat_mode"],
                    "mirostat_tau": config["mirostat_tau"],
                    "mirostat_eta": config["mirostat_eta"]
                })

            # Generate in chunks if response is long
            response = self.model(prompt, **generation_params)
            full_response = response['choices'][0]['text'].strip()

            # If response was cut off, continue generating
            if len(full_response.split()) > 400:  # Approximate token count
                continuation = self.model(
                    prompt + full_response,
                    max_tokens=256,  # Additional tokens
                    stop=generation_params["stop"]
                )
                full_response += " " + continuation['choices'][0]['text'].strip()

            return full_response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    async def generate_response_async(self, prompt: str, **kwargs) -> str:
        """Generate response asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_response, prompt, kwargs)

    def format_streaming_response(self, prompt: str, **kwargs):
        """Generate streaming response for better UX"""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        config = self.model_configs.get(self.model_type, self.model_configs["zephyr"])

        generation_params = {
            "max_tokens": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", config["temperature"]),
            "top_p": kwargs.get("top_p", config["top_p"]),
            "top_k": kwargs.get("top_k", config["top_k"]),
            "repeat_penalty": kwargs.get("repeat_penalty", config["repeat_penalty"]),
            "stop": kwargs.get("stop", ["</s>", "<|user|>", "<|system|>", "\n\nHuman:", "\n\nUser:", "###"]),
            "stream": True,
            "echo": False
        }

        # Add Zephyr-specific parameters
        if self.model_type == "zephyr" and "mirostat_mode" in config:
            generation_params.update({
                "mirostat_mode": config["mirostat_mode"],
                "mirostat_tau": config["mirostat_tau"],
                "mirostat_eta": config["mirostat_eta"]
            })

        return self.model(prompt, **generation_params)


class ChatbotPersonality:
    """Handles chatbot personality and responses"""

    def __init__(self):
        self.greetings = [
            "Hello! I'm Jayoma Bot, your Manila City Ordinance assistant. How can I help you today?",
            "Hi there! I'm here to help you find information about Manila City ordinances. What would you like to know?",
            "Welcome! I'm Jayoma Bot. I can help you search for ordinances, explain penalties, and more. What are you looking for?"
        ]

        self.farewells = [
            "Thank you for using Jayoma Bot! Have a great day!",
            "Goodbye! Feel free to come back if you need more information about ordinances.",
            "Take care! I'm always here if you need help with Manila City ordinances."
        ]

        self.help_messages = [
            "I can help you with:\n• Finding ordinances by topic (e.g., 'smoking ban', 'business permits')\n• Looking up specific ordinance numbers\n• Understanding penalties and fines\n• Getting information about ordinance categories\n\nJust type your question!",
            "Here's what I can do:\n• Search for ordinances by keyword\n• Find all ordinances in a category\n• Provide details about fines and penalties\n• Show ordinance dates and status\n\nWhat would you like to know?"
        ]

        self.clarification_prompts = [
            "I found several ordinances related to '{}'. Could you be more specific about what aspect interests you?",
            "There are multiple '{}' ordinances. Are you looking for something specific like penalties, requirements, or recent updates?",
            "I see you're interested in '{}'. Would you like to see all related ordinances or something more specific?"
        ]

        self.no_results_messages = [
            "I couldn't find any ordinances matching '{}'. Try using different keywords or ask me for help!",
            "No ordinances found for '{}'. Would you like to try a broader search or different terms?",
            "Sorry, I didn't find anything for '{}'. You can try:\n• Using simpler keywords\n• Checking for typos\n• Asking about general categories"
        ]

       # In the ChatbotPersonality class, update the out_of_scope_messages:
        self.out_of_scope_messages = [
            "I'm specifically designed to help with Manila City ordinances. I can't help with '{}', but I can assist you with:\n• Local laws and regulations\n• Business permits and licenses\n• Fines and penalties\n• Tax ordinances\n• Health and safety rules",

            "I'm Jayoma Bot, focused on Manila City ordinances. While I can't answer about '{}', I can help you find:\n• Specific ordinance numbers\n• City regulations (smoking, noise, waste)\n• Permit requirements\n• Local government policies",

            "Sorry, '{}' is outside my expertise. I'm specialized in Manila City ordinances. Try asking about:\n• Business regulations\n• Property ordinances\n• Public health rules\n• Transportation policies\n• Environmental regulations"
        ]

    def get_greeting(self):
        return random.choice(self.greetings)

    def get_farewell(self):
        return random.choice(self.farewells)

    def get_help_message(self):
        return random.choice(self.help_messages)

    def get_clarification(self, topic):
        return random.choice(self.clarification_prompts).format(topic)

    def get_no_results_message(self, query):
        return random.choice(self.no_results_messages).format(query)

    def get_out_of_scope_message(self, query):
        return random.choice(self.out_of_scope_messages).format(query)


class OrdinanceRetrievalSystem:
    def __init__(self, llm_model_path: str = None, llm_model_type: str = "zephyr"):
        """Initialize the retrieval system with optional RAG capabilities and performance optimizations"""
        self.df = None
        self.model = None
        self.cross_encoder = None
        self.embeddings = None
        self.bm25 = None
        self.tokenizer = None
        self.id_to_idx = {}
        self.available_ordinance_ids = set()
        self.chatbot = ChatbotPersonality()
        self.conversation_context = []
        self.last_search_results = []  # Store last search results for "show all"

        # Performance optimizations
        self.faiss_index = None
        self.embedding_cache = {}  # Cache for frequent queries
        self.max_cache_size = 100

        # Simple memory features
        self.user_name = None
        self.search_history = []  # Remember what user searched for
        self.interaction_count = 0  # Count interactions
        self.favorite_topics = Counter()  # Track user interests

        # Load memory from previous session
        self.load_memory()

        self.best_params = {
            'semantic_weight': 0.6,
            'ce_weight': 0.5,
            'batch_size': 8,
            'learning_rate': 3e-5,
            'epochs': 2
        }

        # Initialize RAG components
        self.rag_llm = None
        self.conversation_history = []
        self.max_history_length = 5
        self.use_rag = False

        if llm_model_path:
            try:
                self.rag_llm = RAGOrdinanceLLM(model_path=llm_model_path, model_type=llm_model_type)
                self.use_rag = True
                logger.info(f"RAG mode enabled with {llm_model_type} LLM")
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {str(e)}")
                self.use_rag = False

        # Enhanced query expansion with more comprehensive terms
        self.query_expansion_terms = {
            "covid": ["covid-19", "coronavirus", "pandemic", "quarantine", "lockdown", "health protocol", "face mask", "social distancing", "vaccine", "covid response"],
            "tax": ["levy", "revenue", "assessment", "dues", "collection", "fiscal", "fee", "amusement tax", "real property tax"],
            "property": ["real estate", "land", "building", "house", "lot", "title", "structure", "premises", "estate", "realty"],
            "business": ["enterprise", "commerce", "establishment", "shop", "store", "business permit", "commercial", "firm", "company", "trade"],
            "health": ["sanitation", "hygiene", "medical", "clinic", "cleanliness", "public health", "healthcare", "infection control", "health program"],
            "smoking": ["tobacco", "cigarette", "vape", "e-cigarette", "nicotine", "no smoking", "smoke-free", "secondhand smoke", "smoking ban"],
            "alcohol": ["liquor", "alcoholic drink", "alcoholic beverage", "booze", "beer", "wine", "hard drinks", "intoxicating drink", "drinking ban", "liquor ban"],
            "budget": ["funds", "allocation", "appropriation", "expenditure", "financial", "money", "funding", "fiscal year", "disbursement"],
            "ordinance": ["law", "regulation", "mandate", "decree", "order", "legislation", "policy", "rule", "statute"],
            "penalty": ["fine", "punishment", "sanction", "violation", "offense", "imprisonment", "penalize", "consequence"],
            "permit": ["license", "authorization", "certification", "registration", "approval", "clearance", "franchise"],
            "waste": ["garbage", "trash", "refuse", "rubbish", "disposal", "solid waste", "junk", "basura", "collection"],
            "environment": ["pollution", "air quality", "green", "climate", "clean air", "ecology", "environmental protection"],
            "curfew": ["time restriction", "minor restriction", "night ban", "youth curfew", "curfew hours", "prohibited time"],
            "noise": ["loud sounds", "speakers", "karaoke", "disturbance", "amplifier", "sound pollution", "noise control"]
        }

        # Common chatbot interactions
        self.chatbot_responses = {
            "hello": self.chatbot.get_greeting,
            "hi": self.chatbot.get_greeting,
            "help": self.chatbot.get_help_message,
            "what can you do": self.chatbot.get_help_message,
            "bye": self.chatbot.get_farewell,
            "goodbye": self.chatbot.get_farewell,
            "thank you": lambda: "You're welcome! Is there anything else you'd like to know about ordinances?",
            "thanks": lambda: "You're welcome! Feel free to ask more questions!",
            "how are you": lambda: "I'm doing great! Ready to help you find ordinance information. What can I search for you?",
            "who are you": lambda: "I'm Jayoma Bot, your Manila City Ordinance assistant. I can help you find and understand local ordinances."
        }

        # Enhanced query expansion for RAG
        self.rag_query_templates = {
            "explanation": "Explain what {} means in the context of Manila City ordinances",
            "process": "What is the process for {} according to Manila City ordinances",
            "requirements": "What are the requirements for {} based on Manila City ordinances",
            "penalties": "What are the penalties or fines for {} violations",
            "comparison": "Compare the different ordinances related to {}"
        }

        # Short-term memory for recent ordinances
        self.recent_ordinances = []  # Store last 3 ordinances discussed
        self.max_recent_ordinances = 3

    def remember_ordinance_context(self, ordinance: Dict):
        """Remember recently discussed ordinances for better context"""
        if ordinance.get('ordinance_id') != 'N/A':
            # Remove if already exists to avoid duplicates
            self.recent_ordinances = [o for o in self.recent_ordinances
                                    if o['ordinance_id'] != ordinance['ordinance_id']]
            # Add to front
            self.recent_ordinances.insert(0, {
                'ordinance_id': ordinance['ordinance_id'],
                'category': ordinance.get('category', ''),
                'topic': ordinance.get('short_text', '')[:100]
            })
            # Keep only last N ordinances
            self.recent_ordinances = self.recent_ordinances[:self.max_recent_ordinances]

    def get_search_guidance(self, query: str) -> str:
        """Provide helpful guidance for failed searches"""
        query_lower = query.lower()

        # Common search patterns and suggestions
        if any(word in query_lower for word in ['e-trike', 'etrike', 'electric', 'tricycle']):
            return "💡 Try searching for:\n• 'transport subsidies'\n• 'public transit ordinances'\n• 'tricycle modernization'\n• 'transport assistance programs'"

        elif any(word in query_lower for word in ['free', 'subsidy', 'assistance']):
            return "💡 Try searching for:\n• 'financial assistance'\n• 'subsidy programs'\n• 'social services'\n• 'government aid ordinances'"

        elif any(word in query_lower for word in ['restaurant', 'food', 'dining']):
            return "💡 Try searching for:\n• 'food establishment permits'\n• 'restaurant licenses'\n• 'health permits for food'\n• 'sanitation requirements'"

        else:
            # General guidance
            return "💡 Try:\n• Using simpler keywords\n• Searching for related terms\n• Asking about specific aspects (permits, fines, requirements)\n• Type 'help' to see what I can do"

    def generate_follow_up_questions(self, results: List[Dict], intent: Dict) -> str:
        """Generate contextual follow-up questions based on results"""
        if not results:
            return ""

        follow_ups = []

        # Based on intent
        if intent['type'] == 'general':
            # Check what information is available
            has_fines = any(r.get('fines') and r['fines'] != 'N/A' for r in results)
            has_dates = any(r.get('date_enacted') and r['date_enacted'] != 'Date not available' for r in results)

            if has_fines:
                follow_ups.append("see the fines for any of these ordinances")
            if has_dates:
                follow_ups.append("know when these were enacted")
            if len(results) > 1:
                follow_ups.append("get more details about a specific ordinance")

        elif intent['type'] == 'fines':
            follow_ups.append("see the full ordinance text")

        elif intent['type'] == 'details':
            follow_ups.append("see related ordinances")
            follow_ups.append("know the penalties for violations")

        # Add context from recent ordinances
        if self.recent_ordinances and len(self.recent_ordinances) > 1:
            recent = self.recent_ordinances[1]  # Get second most recent (first is current)
            follow_ups.append(f"go back to Ordinance {recent['ordinance_id']} about {recent['category']}")

        if follow_ups:
            return "Would you like to " + " or ".join(follow_ups) + "?"

        return ""

    def analyze_query_specificity(self, query: str) -> Tuple[str, List[str]]:
        """Analyze if query is general or specific and extract key terms"""
        query_lower = query.lower()
        tokens = set(re.findall(r'\w+', query_lower))

        # Extract main topic and modifiers
        main_topics = []
        modifiers = []

        # Check for expanded terms
        for term, expansions in self.query_expansion_terms.items():
            if term in tokens or any(exp in query_lower for exp in expansions):
                main_topics.append(term)

        # Common specific modifiers that indicate user wants filtered results
        specific_indicators = ['budget', 'penalty', 'fine', 'requirement', 'application',
                             'process', 'fee', 'cost', 'violation', 'enforcement',
                             'specific', 'particular', 'related to', 'about', 'regarding']

        for indicator in specific_indicators:
            if indicator in query_lower:
                modifiers.append(indicator)

        # Determine query type
        if len(tokens) <= 2 and len(modifiers) == 0:
            query_type = "general"
        elif len(modifiers) > 0 or len(tokens) > 5:
            query_type = "specific"
        else:
            query_type = "moderate"

        return query_type, main_topics

    def is_out_of_scope(self, query: str) -> bool:
        """DEPRECATED - Out of scope detection is now handled by score threshold in retrieve_ordinances"""
        return False

    def save_memory(self):
        """Save user memory to a JSON file"""
        memory_data = {
            'user_name': self.user_name,
            'search_history': self.search_history[-20:],  # Keep last 20 searches
            'interaction_count': self.interaction_count,
            'favorite_topics': dict(self.favorite_topics.most_common(10))  # Top 10 topics
        }
        try:
            with open('chatbot_memory.json', 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save memory: {str(e)}")

    def load_memory(self):
        """Load user memory from previous session"""
        try:
            with open('chatbot_memory.json', 'r') as f:
                memory_data = json.load(f)
                self.user_name = memory_data.get('user_name')
                self.search_history = memory_data.get('search_history', [])
                self.interaction_count = memory_data.get('interaction_count', 0)
                self.favorite_topics = Counter(memory_data.get('favorite_topics', {}))
        except FileNotFoundError:
            logger.info("No previous memory found, starting fresh")
        except Exception as e:
            logger.warning(f"Could not load memory: {str(e)}")

    def remember_search(self, query, results):
        """Remember what the user searched for"""
        # Add to search history
        self.search_history.append({
            'query': query,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results_count': len(results) if isinstance(results, list) else 0
        })

        # Track topics
        query_lower = query.lower()
        for topic in self.query_expansion_terms.keys():
            if topic in query_lower:
                self.favorite_topics[topic] += 1

        # Save memory periodically
        self.interaction_count += 1
        if self.interaction_count % 5 == 0:  # Save every 5 interactions
            self.save_memory()

    def get_personalized_greeting(self):
        """Get a personalized greeting based on memory"""
        if self.user_name:
            base_greeting = f"Welcome back, {self.user_name}! "
        else:
            base_greeting = ""

        if self.favorite_topics and self.interaction_count > 5:
            top_topic = self.favorite_topics.most_common(1)[0][0]
            return base_greeting + f"I remember you're interested in {top_topic} ordinances. How can I help you today?"
        elif self.search_history and self.interaction_count > 0:
            last_search = self.search_history[-1]['query']
            return base_greeting + f"Last time you searched for '{last_search}'. What would you like to know today?"
        else:
            return self.chatbot.get_greeting()

    def extract_query_intent(self, query: str) -> Dict[str, any]:
        """Extract what specific information the user is asking for"""
        query_lower = query.lower()

        intent = {
            'type': None,  # 'fines', 'date', 'details', 'category', 'general'
            'topic': None,
            'specific_field': None
        }

        # Check what information they're asking for
        if any(word in query_lower for word in ['fine', 'fines', 'penalty', 'penalties', 'how much', 'cost', 'fee']):
            intent['type'] = 'fines'
        elif any(word in query_lower for word in ['when', 'date', 'enacted', 'passed', 'approved']):
            intent['type'] = 'date'
        elif any(word in query_lower for word in ['category', 'type', 'classification']):
            intent['type'] = 'category'
        elif any(word in query_lower for word in ['what is', 'explain', 'describe', 'tell me about']):
            intent['type'] = 'details'
        else:
            intent['type'] = 'general'

        # Extract the topic
        for term in self.query_expansion_terms.keys():
            if term in query_lower:
                intent['topic'] = term
                break

        return intent



    def format_conversational_response(self, results: List[Dict], intent: Dict) -> str:
        """Format the response in a conversational way based on intent"""
        if not results or (isinstance(results, list) and len(results) == 0):
            return "I couldn't find any ordinances related to your query. Could you try asking differently?"

        # Handle direct list results
        if isinstance(results, list) and results[0].get('ordinance_id') != 'N/A':
            # Get the top 1-2 most relevant results
            top_results = results[:2]

            if intent['type'] == 'fines':
                # User asking about fines/penalties
                responses = []
                for res in top_results:
                    if res.get('fines') and res['fines'] != "" and res['fines'] != "N/A":
                        responses.append(f"According to Ordinance #{res['ordinance_id']}, the fines are: {res['fines']}")

                if responses:
                    if len(responses) == 1:
                        return responses[0]
                    else:
                        return "Here are the fines I found:\n\n" + "\n\n".join(responses)
                else:
                    return f"I found relevant ordinances, but the enactment dates are not available in the database."

            elif intent['type'] == 'category':
                # User asking about category/classification
                res = top_results[0]
                return f"Ordinance #{res['ordinance_id']} falls under the category of '{res.get('category', 'Unknown')}'."

            elif intent['type'] == 'details':
                # User wants explanation/details
                res = top_results[0]
                response = f"Ordinance #{res['ordinance_id']} states: {res['short_text']}"
                if res.get('fines') and res['fines'] != "" and res['fines'] != "N/A":
                    response += f"\n\nFines: {res['fines']}"
                if res.get('date_enacted') and res['date_enacted'] != "Date not available":
                    response += f"\n\nThis was enacted on {res['date_enacted']}."
                return response

            else:
                # General query - show summary
                res = top_results[0]
                response = f"I found Ordinance #{res['ordinance_id']} about {intent['topic'] or 'your query'}."
                response += f"\n\n{res['short_text']}"
                if len(top_results) > 1:
                    response += f"\n\nThere's also Ordinance #{top_results[1]['ordinance_id']} that might be relevant."
                return response

        return "I'm having trouble understanding your question. Could you please rephrase it?"

    def is_chatbot_interaction(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if the query is a chatbot interaction with improved matching"""
        query_lower = query.lower().strip()

        # Check for exact matches first
        exact_matches = {
            "hi": self.chatbot.get_greeting,
            "hello": self.chatbot.get_greeting,
            "help": self.chatbot.get_help_message,
            "what can you do": self.chatbot.get_help_message,
            "bye": self.chatbot.get_farewell,
            "goodbye": self.chatbot.get_farewell,
            "thank you": lambda: "You're welcome! Is there anything else you'd like to know about ordinances?",
            "thanks": lambda: "You're welcome! Feel free to ask more questions!",
        }

        # Check for exact matches
        if query_lower in exact_matches:
            return True, exact_matches[query_lower]()

        # Check for "show all" variations
        if query_lower in ["show all", "view all", "see all", "display all"]:
            return True, "show_all"

        # Check for name introduction with regex
        name_pattern = r'^(?:my name is|i am|call me)\s+(\w+)$'
        if re.match(name_pattern, query_lower):
            name = re.match(name_pattern, query_lower).group(1).capitalize()
            self.user_name = name
            self.save_memory()
            return True, f"Nice to meet you, {self.user_name}! How can I help you with Manila City ordinances?"

        # Check for history request
        history_phrases = [
            "what did i search",
            "my history",
            "previous searches",
            "search history"
        ]
        if any(phrase in query_lower for phrase in history_phrases):
            if self.search_history:
                recent = self.search_history[-5:]
                history_text = "Here are your recent searches:\n"
                for i, search in enumerate(recent, 1):
                    history_text += f"{i}. '{search['query']}' - {search['timestamp']}\n"
                return True, history_text
            return True, "You haven't searched for anything yet."

        return False, None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the ordinance data with robust error handling"""
        try:
            logger.info(f"Loading data from {file_path}...")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found at {file_path}")

            dtype_mapping = {
                'ordinance_id': str,
                'short_text': str,
                'full_text': str,
                'category': str,
                'fines': str,
                'date_enacted': str,
                'status': str,
                'links': str
            }

            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        file_path,
                        na_values=["nan", "NaN", "NULL", "None", "MISSING", "TOO LONG", ""],
                        dtype=dtype_mapping,
                        encoding=encoding
                    )
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read with {encoding} encoding, trying next...")
                    continue
                except Exception as e:
                    logger.error(f"Failed to read CSV with {encoding}: {str(e)}")
                    continue

            if df is None:
                raise ValueError("Failed to read CSV with any supported encoding")

            required_columns = ['ordinance_id', 'short_text']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Clean special characters that might cause issues
            def clean_text(text):
                if pd.isna(text):
                    return ""
                # Replace common problematic characters
                text = str(text)
                replacements = {
                    '\x92': "'",  # Windows-1252 apostrophe
                    '\x93': '"',  # Left double quotation mark
                    '\x94': '"',  # Right double quotation mark
                    '\x96': '-',  # En dash
                    '\x97': '-',  # Em dash
                    '\xa0': ' ',  # Non-breaking space
                    '\u2019': "'",  # Right single quotation mark
                    '\u201c': '"',  # Left double quotation mark
                    '\u201d': '"',  # Right double quotation mark
                    '\u2013': '-',  # En dash
                    '\u2014': '-',  # Em dash
                }
                for old, new in replacements.items():
                    text = text.replace(old, new)
                # Remove any remaining non-ASCII characters
                text = text.encode('ascii', 'ignore').decode('ascii')
                return text

            text_cols = ["short_text", "full_text", "category", "fines", "status", "links"]
            for col in text_cols:
                if col in df.columns:
                    try:
                        df[col] = df[col].fillna("").apply(clean_text)
                        df[col] = df[col].apply(lambda x: " ".join(str(x).split()) if pd.notna(x) else "")
                    except Exception as e:
                        logger.warning(f"Error cleaning column {col}: {str(e)}")
                        df[col] = df[col].astype(str).fillna("")

            if "category" in df.columns:
                df["category"] = df["category"].replace("", "Unknown").fillna("Unknown")

            if "status" in df.columns:
                df["status"] = df["status"].replace("", "Status not specified").fillna("Status not specified")

            if "date_enacted" in df.columns:
                try:
                    df["date_enacted"] = pd.to_datetime(
                        df["date_enacted"],
                        errors="coerce",
                        format='mixed'
                    ).dt.strftime('%B %d, %Y')
                    df["date_enacted"] = df["date_enacted"].fillna("Date not available")
                except Exception as e:
                    logger.warning(f"Error parsing dates: {str(e)}")
                    df["date_enacted"] = "Date not available"

            if "short_text" in df.columns:
                df = df[df["short_text"].str.len() > 30].copy()

            self.df = df.reset_index(drop=True)
            self.available_ordinance_ids = set(self.df["ordinance_id"].astype(str).unique())
            logger.info(f"Successfully loaded {len(self.df)} ordinances")
            return self.df

        except Exception as e:
            logger.error(f"Critical error in load_data: {str(e)}")
            raise

    def initialize_models(self):
        """Initialize the retrieval models with error handling"""
        try:
            logger.info("Initializing models...")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
                logger.info("Tokenizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load legal-bert tokenizer, falling back to default: {str(e)}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    logger.info("Fallback tokenizer initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize any tokenizer: {str(e)}")
                    self.tokenizer = None

            try:
                self.model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
            except Exception as e:
                logger.warning(f"Failed to load legal-bert, falling back to all-MiniLM: {str(e)}")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {str(e)}")
                self.cross_encoder = None

            try:
                if self.df is not None and len(self.df) > 0 and self.tokenizer is not None:
                    tokenized_corpus = [self.tokenizer.tokenize(str(text)) for text in self.df["short_text"]]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.info("BM25 initialized successfully")
                else:
                    logger.warning("No data or tokenizer available for BM25 initialization")
                    self.bm25 = None
            except Exception as e:
                logger.error(f"Failed to initialize BM25: {str(e)}")
                self.bm25 = None

            logger.info("Models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def generate_embeddings(self):
        """Generate embeddings for all ordinances with enhanced context"""
        try:
            logger.info("Generating embeddings...")

            if self.model is None:
                raise ValueError("Model not initialized")

            if len(self.df) == 0:
                raise ValueError("No data available for embedding generation")

            # Create richer text representations for better semantic matching
            texts = []
            for _, row in self.df.iterrows():
                # Combine multiple fields for richer context
                text_parts = [
                    str(row["short_text"]),
                    f"Category: {row.get('category', '')}",
                    f"Fines: {row.get('fines', '')}" if row.get('fines', '') else "",
                    f"Status: {row.get('status', '')}" if row.get('status', '') else ""
                ]
                combined_text = " ".join(filter(None, text_parts))
                texts.append(combined_text)

            chunk_size = 100
            embeddings = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                try:
                    embeddings.append(self.model.encode(chunk, show_progress_bar=False))
                except Exception as e:
                    logger.warning(f"Error encoding chunk {i//chunk_size}: {str(e)}")
                    raise

            self.embeddings = np.concatenate(embeddings)
            self.id_to_idx = {str(id): idx for idx, id in enumerate(self.df["ordinance_id"])}

            logger.info(f"Generated embeddings for {len(self.embeddings)} ordinances")

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def expand_query(self, query: str) -> str:
        """Enhanced query expansion with better synonym integration"""
        try:
            query_lower = query.lower()
            expanded_terms = set(query_lower.split())

            # Add synonyms for each matching term
            for term, synonyms in self.query_expansion_terms.items():
                if term in query_lower:
                    # Add the most relevant synonyms
                    expanded_terms.update(synonyms[:5])

                # Also check if any synonym appears in the query
                for synonym in synonyms:
                    if synonym in query_lower and term not in expanded_terms:
                        expanded_terms.add(term)
                        expanded_terms.update(synonyms[:3])
                        break

            return " ".join(expanded_terms)
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return query


    def is_query_out_of_scope(self, query: str, results: List[Dict] = None, max_score: float = None) -> bool:
        """
        Enhanced out-of-scope detection with multiple checks
        """
        query_lower = query.lower()

        # 1. Keyword-based filtering for obviously out-of-scope queries
        out_of_scope_patterns = [
            # Gaming
            r'\b(dota|league of legends|minecraft|fortnite|pubg|valorant|game|gaming|play station|xbox|nintendo)\b',
            # Cooking/Food (unless related to permits)
            r'\b(cook|recipe|fried chicken|adobo|sinigang|pasta|pizza|burger|restaurant menu)\b',
            # Animals (unless related to ordinances)
            r'\b(cat|dog|pet|animal|bird|fish)\b(?!.*(?:ordinance|permit|registration|license|violation))',
            # Entertainment
            r'\b(movie|film|music|song|concert|netflix|youtube|tiktok)\b(?!.*(?:permit|license|ordinance|tax))',
            # Technical/Programming (unless ordinance-related)
            r'\b(python|java|programming|coding|algorithm|software|hardware)\b(?!.*(?:ordinance|system))',
            # General knowledge
            r'\b(weather|temperature|climate|geography|history|science|math)\b(?!.*(?:ordinance|manila|city))',
            # Personal/Social
            r'\b(love|relationship|dating|marriage|friendship)\b(?!.*(?:license|permit|ordinance))',
            # Educational / Academic Questions
            r'\b(math|mathematics|algebra|geometry|calculus|trigonometry|solve|equation|fraction|formula|derivative|integral|probability)\b(?!.*(?:ordinance|manila|policy))',
            r'\b(science|physics|chemistry|biology|experiment|atom|gravity|newton|molecule|mass|velocity|boiling point|photosynthesis)\b(?!.*(?:ordinance|regulation|city))',
            #regex for math
            r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$',

        ]

        for pattern in out_of_scope_patterns:
            if re.search(pattern, query_lower):
                return True

        # 2. Check if query contains ANY ordinance-related keywords
        ordinance_keywords = [
            'ordinance', 'law', 'regulation', 'permit', 'license', 'fine', 'penalty',
            'manila', 'city', 'municipal', 'barangay', 'violation', 'compliance',
            'business', 'tax', 'property', 'health', 'sanitation', 'waste', 'smoke',
            'smoking', 'alcohol', 'curfew', 'noise', 'transport', 'tricycle', 'jeepney',
            'vendor', 'market', 'establishment', 'fee', 'payment', 'requirement',
            'application', 'certificate', 'clearance', 'registration', 'policy'
        ]

        has_ordinance_keyword = any(keyword in query_lower for keyword in ordinance_keywords)

        # 3. Score-based check (if we have results)
        if max_score is not None and max_score < 0.15:
            return True

        # 4. Combined decision
        if not has_ordinance_keyword and len(query_lower.split()) > 2:
            # Multi-word query with no ordinance keywords = likely out of scope
            return True

        return False

    def is_ordinance_related(self, query: str) -> bool:
        """Check if query is definitely ordinance-related"""
        query_lower = query.lower()

        # Direct ordinance references
        if re.search(r'ordinance\s*#?\s*\d+', query_lower):
            return True

        # Ordinance-specific phrases
        ordinance_phrases = [
            'manila city ordinance',
            'city ordinance',
            'local ordinance',
            'municipal law',
            'city regulation',
            'manila law',
            'barangay ordinance',
            'city policy',
            'municipal code'
        ]

        return any(phrase in query_lower for phrase in ordinance_phrases)




    def get_all_matching_ordinances(self, keyword: str) -> List[Dict]:
        """Get all ordinances that contain a specific keyword"""
        matching_ordinances = []
        keyword_lower = keyword.lower()

        for idx, row in self.df.iterrows():
            # Check if keyword appears in any relevant field
            text_to_search = f"{row['short_text']} {row.get('category', '')} {row.get('fines', '')}".lower()

            if keyword_lower in text_to_search:
                matching_ordinances.append({
                    "ordinance_id": row["ordinance_id"],
                    "category": row.get("category", ""),
                    "short_text": row["short_text"],
                    "fines": row.get("fines", ""),
                    "date_enacted": row.get("date_enacted", ""),
                    "status": row.get("status", "Status not specified"),
                    "links": row.get("links", "Links not specified"),
                    "confidence": "Keyword Match",
                    "score": "Direct Match"
                })

        return matching_ordinances

    def retrieve_ordinances(self, query: str, k: int = 5,
                          semantic_weight: float = None,
                          ce_weight: float = None) -> Union[List[Dict], str, Tuple]:
        """Enhanced retrieve_ordinances that can return RAG responses"""
        try:
            # Check if it's a chatbot interaction first
            is_chat, chat_response = self.is_chatbot_interaction(query)
            if is_chat:
                if chat_response == "show_all":
                    # Return all stored results
                    if self.last_search_results:
                        return "show_all_results", self.last_search_results
                    else:
                        return "No previous search results to show. Please search for something first!", []
                return chat_response

            # EARLY OUT-OF-SCOPE CHECK - before any processing
            if self.is_query_out_of_scope(query):
                return self.chatbot.get_out_of_scope_message(query), []

            # Extract intent from query
            intent = self.extract_query_intent(query)

            # Analyze query specificity
            query_type, main_topics = self.analyze_query_specificity(query)

            # For very general single-word queries (e.g., just "COVID")
            if query_type == "general" and len(main_topics) == 1 and intent['type'] == 'general':
                logger.info(f"General query detected for topic: {main_topics[0]}")

                # Instead of auto-searching, ask for clarification
                suggestions = []
                topic = main_topics[0]

                if topic == "covid":
                    suggestions = [
                        "COVID-19 funds",
                        "COVID-19 health protocols",
                        "COVID-19 vaccine ordinances",
                        "COVID-19 business restrictions",
                        "quarantine facility regulations",
                        "COVID-19 assistance"
                    ]

                elif topic == "business":
                    suggestions = [
                        "business permits",
                        "business license requirements",
                        "business operating hours",
                        "business tax",
                        "market vendor regulations",
                        "commercial establishment rules",
                        "business fee exemptions",
                        "trade and commerce policies"
                    ]

                elif topic == "tax":
                    suggestions = [
                        "property tax",
                        "business tax",
                        "amusement tax",
                        "tax penalties",
                        "tax exemptions",
                        "tax amnesty programs",
                        "real property tax updates"
                    ]

                elif topic == "smoking":
                    suggestions = [
                        "smoking ban locations",
                        "smoking fines",
                        "vaping regulations",
                        "smoke-free establishments",
                        "tobacco control measures",
                        "designated smoking areas"
                    ]

                elif topic == "transportation":
                    suggestions = [
                        "tricycle route regulations",
                        "parking regulations",
                        "traffic management",
                        "loading/unloading zones",
                        "public transport operators",
                        "vehicle registration",
                        "terminal operations",
                        "jeepney routes and franchises"
                    ]

                elif topic == "housing":
                    suggestions = [
                        "land acquisition programs",
                        "Land-for-the-Landless",
                        "housing assistance",
                        "property purchase authority",
                        "tenant qualification",
                        "resettlement programs",
                        "urban development",
                        "affordable housing initiatives"
                    ]

                elif topic == "health":
                    suggestions = [
                        "health center operations",
                        "medical assistance programs",
                        "hospital services",
                        "health emergency response",
                        "public health protocols",
                        "medical personnel positions",
                        "health facility improvements",
                        "disease prevention programs"
                    ]

                elif topic == "education":
                    suggestions = [
                        "scholarship programs",
                        "school infrastructure",
                        "educational assistance",
                        "student welfare",
                        "teacher positions",
                        "educational facilities",
                        "youth development programs"
                    ]

                elif topic == "environment":
                    suggestions = [
                        "waste management",
                        "garbage collection",
                        "recycling programs",
                        "environmental protection",
                        "solid waste disposal",
                        "clean-up operations",
                        "green initiatives"
                    ]

                elif topic == "welfare":
                    suggestions = [
                        "senior citizen benefits",
                        "social services programs",
                        "disability assistance",
                        "welfare fund allocation",
                        "subsidy programs",
                        "social assistance",
                        "vulnerable sector support",
                        "community welfare initiatives"
                    ]

                elif topic == "permits":
                    suggestions = [
                        "mayor's permit fees",
                        "permit exemptions",
                        "license requirements",
                        "permit applications",
                        "certificate issuance",
                        "registration procedures",
                        "fee waivers"
                    ]

                elif topic == "culture":
                    suggestions = [
                        "cultural events",
                        "festival celebrations",
                        "tourism initiatives",
                        "museum operations",
                        "heritage preservation",
                        "arts programs",
                        "cultural facility management",
                        "event permits"
                    ]

                elif topic == "safety":
                    suggestions = [
                        "public safety measures",
                        "security protocols",
                        "emergency response",
                        "crime prevention",
                        "barangay peacekeeping",
                        "disaster preparedness",
                        "fire safety regulations",
                        "law enforcement"
                    ]

                elif topic == "government":
                    suggestions = [
                        "government positions",
                        "salary grades",
                        "department creation",
                        "budget amendments",
                        "organizational structure",
                        "employee benefits",
                        "administrative orders",
                        "government operations"
                    ]

                elif topic == "infrastructure":
                    suggestions = [
                        "building construction",
                        "road improvements",
                        "facility development",
                        "infrastructure projects",
                        "public works",
                        "construction permits",
                        "facility maintenance"
                    ]

                elif topic == "youth":
                    suggestions = [
                        "youth programs",
                        "children welfare",
                        "sports development",
                        "recreation facilities",
                        "youth organizations",
                        "child protection",
                        "student assistance"
                    ]

                elif topic == "senior":
                    suggestions = [
                        "senior citizen ID",
                        "elderly care programs",
                        "senior discounts",
                        "pension benefits",
                        "elderly health services",
                        "senior activity centers"
                    ]

                elif topic == "employment":
                    suggestions = [
                        "worker benefits",
                        "employee compensation",
                        "job order positions",
                        "labor regulations",
                        "contract workers",
                        "salary adjustments",
                        "employment programs"
                    ]

                elif topic == "fees":
                    suggestions = [
                        "fee schedules",
                        "fee exemptions",
                        "payment procedures",
                        "penalty charges",
                        "service fees",
                        "administrative charges",
                        "fee waivers"
                    ]
                else:
                    suggestions = [f"{topic} permits", f"{topic} fines", f"{topic} requirements", f"{topic} regulations"]

                response = f"Your search for '{topic}' is quite broad. Did you mean:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    response += f"  {i}. {suggestion}\n"
                response += f"\nPlease be more specific with the queries, Thank you!"

                # Store the suggestions for quick selection
                self.last_suggestions = suggestions
                return response

            # Check if user selected a suggestion number
            if query.isdigit() and hasattr(self, 'last_suggestions'):
                try:
                    choice = int(query) - 1
                    if 0 <= choice < len(self.last_suggestions):
                        query = self.last_suggestions[choice]
                        # Clear suggestions after use
                        delattr(self, 'last_suggestions')
                except:
                    pass

            # Check for ordinance ID query
            if self._is_ordinance_id_query(query):
                ordinance_id = self._extract_ordinance_id(query)
                if ordinance_id:
                    ordinance = self._get_ordinance_by_id(ordinance_id)
                    if ordinance:
                        # Remember this ordinance for context
                        self.remember_ordinance_context(ordinance)
                        return [ordinance]
                    else:
                        return [{
                            "ordinance_id": ordinance_id,
                            "category": "N/A",
                            "short_text": f"Ordinance {ordinance_id} data entry is MISSING from our records.",
                            "fines": "N/A",
                            "date_enacted": "N/A",
                            "status": "N/A",
                            "links": "N/A",
                            "confidence": "Not Found",
                            "score": "0.0%"
                        }]

            # Use best params if none provided
            semantic_weight = semantic_weight or self.best_params['semantic_weight']
            ce_weight = ce_weight or self.best_params['ce_weight']

            # Enhanced query expansion
            expanded_query = self.expand_query(query)

            # Generate query embedding with context
            try:
                # For specific queries, emphasize the specific aspects
                if query_type == "specific":
                    context_query = f"{query} specific details requirements"
                else:
                    context_query = query

                query_embs = self.model.encode([context_query, expanded_query], convert_to_tensor=True)
                query_emb = torch.mean(query_embs, dim=0).cpu().numpy()

                tokenized_query = self.tokenizer.tokenize(query.lower())
                expanded_tokens = self.tokenizer.tokenize(expanded_query.lower())
                all_tokens = tokenized_query * 2 + expanded_tokens

                logger.info(f"Query type: {query_type}")
                logger.info(f"Original query: {query}")
                logger.info(f"Expanded query: {expanded_query}")

            except Exception as e:
                logger.warning(f"Query processing failed: {str(e)}")
                query_emb = self.model.encode(query, convert_to_tensor=True).cpu().numpy()
                all_tokens = self.tokenizer.tokenize(query.lower())

            # Calculate semantic scores with improved similarity
            semantic_scores = cosine_similarity([query_emb], self.embeddings)[0]

            # ENHANCED: Boost scores for exact keyword matches
            query_words = set(query.lower().split())
            for idx, row in self.df.iterrows():
                text_lower = row['short_text'].lower()
                category_lower = row.get('category', '').lower()

                # Exact phrase matching bonus
                if query.lower() in text_lower:
                    semantic_scores[idx] *= 2.0  # Double score for exact phrase

                # Boost score if main topics appear in the text
                for topic in main_topics:
                    if topic in text_lower or topic in category_lower:
                        semantic_scores[idx] *= 1.5  # 50% boost for topic match

                # Additional boost for multiple keyword matches
                text_words = set(text_lower.split())
                common_words = query_words.intersection(text_words)
                if len(common_words) > 1:
                    semantic_scores[idx] *= (1 + 0.2 * len(common_words))  # Increased boost

            semantic_scores_norm = self.robust_scale(semantic_scores)

            # Enhanced BM25 scoring
            keyword_scores = np.zeros(len(semantic_scores_norm))
            if self.tokenizer is not None and self.bm25 is not None:
                try:
                    text_scores = self.bm25.get_scores(all_tokens)

                    if 'category' in self.df.columns:
                        category_bm25 = BM25Okapi([self.tokenizer.tokenize(str(cat).lower())
                                                for cat in self.df['category']])
                        category_scores = category_bm25.get_scores(all_tokens)
                        keyword_scores = 0.7 * text_scores + 0.3 * category_scores
                    else:
                        keyword_scores = text_scores

                    keyword_scores_norm = self.robust_scale(keyword_scores)
                except Exception as e:
                    logger.warning(f"Keyword search failed: {str(e)}")
                    keyword_scores_norm = np.zeros(len(semantic_scores_norm))

            # Get combined scores
            if query_type == "specific":
                combined_scores = semantic_scores_norm * 0.7 + keyword_scores_norm * 0.3
            else:
                combined_scores = semantic_scores_norm * 0.6 + keyword_scores_norm * 0.4

            # Check if query is out of scope based on max score
            max_score = np.max(combined_scores) if len(combined_scores) > 0 else 0
            if self.is_query_out_of_scope(query, max_score=max_score):
                guidance = self.get_search_guidance(query)
                return f"{self.chatbot.get_out_of_scope_message(query)}\n\n{guidance}", []

            # For intent-based queries, we want top 2 results max
            if intent['type'] in ['fines', 'date', 'category', 'details']:
                k = 2
            else:
                # Adjust k based on query type
                if query_type == "general":
                    k = min(k * 3, 15)  # Show more results for general queries
                elif query_type == "moderate":
                    k = min(k * 2, 10)  # Moderate number of results

            # Get top candidates with better threshold
            if query_type == "specific":
                threshold = 0.5  # Increased for better precision
            else:
                threshold = 0.3

            # Filter by threshold and get top results
            valid_indices = np.where(combined_scores > threshold)[0]
            if len(valid_indices) == 0:
                valid_indices = np.argsort(combined_scores)[::-1][:k]
            else:
                valid_indices = valid_indices[np.argsort(combined_scores[valid_indices])[::-1]][:k]

            # Cross-encoder re-ranking for specific queries
            if self.cross_encoder and query_type == "specific" and len(valid_indices) > 0:
                try:
                    pairs = []
                    for idx in valid_indices:
                        row = self.df.iloc[idx]
                        context = f"Category: {row.get('category', '')}. "
                        context += f"Text: {row['short_text']}. "
                        if pd.notna(row.get('fines', None)):
                            context += f"Fines: {row['fines']}. "
                        pairs.append((expanded_query, context))

                    ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

                    # Re-sort based on cross-encoder scores
                    ce_indices = np.argsort(ce_scores)[::-1]
                    valid_indices = valid_indices[ce_indices]
                except Exception as e:
                    logger.warning(f"Cross-encoder failed: {str(e)}")

            # Prepare results
            results = []
            for idx in valid_indices:
                try:
                    row = self.df.iloc[idx]
                    score = combined_scores[idx] * 100

                    result = {
                        "ordinance_id": row["ordinance_id"],
                        "category": row.get("category", ""),
                        "short_text": row["short_text"],
                        "fines": row.get("fines", ""),
                        "date_enacted": row.get("date_enacted", ""),
                        "status": row.get("status", "Status not specified"),
                        "links": row.get("links", "Links not specified"),
                        "confidence": self._get_confidence_level(score),
                        "score": f"{score:.1f}%"
                    }
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error formatting result {idx}: {str(e)}")
                    continue

            if not results:
                guidance = self.get_search_guidance(query)
                return f"{self.chatbot.get_no_results_message(query)}\n\n{guidance}", []

            # Store results for potential "show all" command
            self.last_search_results = results

            # Remember this search and context
            self.remember_search(query, results)
            for result in results[:3]:  # Remember top 3 ordinances
                self.remember_ordinance_context(result)

            # If RAG is enabled, generate RAG response
            if self.use_rag and results:
                rag_response = self._generate_rag_response(query, results, intent)

                # Add proactive follow-up questions
                follow_up = self.generate_follow_up_questions(results, intent)
                if follow_up:
                    rag_response += f"\n\n{follow_up}"

                return rag_response, results

            # Otherwise, format conversational response based on intent
            if intent['type'] != 'general':
                conversational_response = self.format_conversational_response(results, intent)

                # Add follow-up questions
                follow_up = self.generate_follow_up_questions(results, intent)
                if follow_up:
                    conversational_response += f"\n\n{follow_up}"

                return conversational_response

            # Add contextual message based on query type
            if query_type == "general" and len(results) > 5:
                intro_message = f"I found {len(results)} ordinances related to your query. Here are the most relevant ones."

                # Add follow-up questions
                follow_up = self.generate_follow_up_questions(results, intent)
                if follow_up:
                    intro_message += f"\n\n{follow_up}"

                return intro_message, results
            else:
                return results

        except Exception as e:
            logger.error(f"Error in retrieve_ordinances: {str(e)}")
            return "I'm sorry, I encountered an error while searching. Please try again with different keywords.", []


    def _generate_rag_response(self, query: str, context: List[Dict], intent: Dict) -> str:
        """Generate RAG response using the LLM with streaming"""
        try:
            # Create the prompt
            prompt = self.rag_llm.create_rag_prompt(
                query=query,
                context=context,
                intent=intent,
                conversation_history=self.conversation_history[-1:] if self.conversation_history else None
            )

            # Don't print here - just collect the response
            response_text = ""
            stream = self.rag_llm.format_streaming_response(prompt)

            for output in stream:
                token = output['choices'][0]['text']
                response_text += token

                # Check for early stopping
                if any(stop in response_text for stop in ["</s>", "<|user|>", "<|system|>"]):
                    break

            # Update conversation history
            self.conversation_history.append({
                'user': query,
                'assistant': response_text.strip(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Trim history if too long
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]

            return response_text.strip()

        except Exception as e:
            logger.error(f"RAG generation failed: {str(e)}")
            # Fallback to basic response
            return self.format_conversational_response(context, intent)

    def _is_ordinance_id_query(self, query: str) -> bool:
        """Check if the query is specifically looking for an ordinance ID"""
        # More flexible patterns to catch various ways users might ask for specific ordinances
        patterns = [
            r'ordinance\s*#?\s*(\d+)',  # ordinance #123, ordinance 123
            r'#\s*(\d+)',                # #123
            r'number\s*#?\s*(\d+)',      # number #123, number 123
            r'no\.?\s*#?\s*(\d+)',       # no. 123, no #123
            r'give\s+me\s+ordinance\s*#?\s*(\d+)',  # give me ordinance #123
            r'show\s+me\s+ordinance\s*#?\s*(\d+)',  # show me ordinance #123
            r'what\s+is\s+ordinance\s*#?\s*(\d+)',  # what is ordinance #123
            r'(?:fines?|penalty|penalties)\s+for\s+(?:ordinance\s*)?#?\s*(\d+)',  # fines for ordinance #123
            r'when\s+was\s+(?:ordinance\s*)?#?\s*(\d+)',  # when was ordinance #123
            r'date\s+.*?(?:ordinance\s*)?#?\s*(\d+)',  # date enacted for ordinance #123
        ]

        query_lower = query.lower()
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _extract_ordinance_id(self, query: str) -> str:
        """Extract the ordinance ID from the query"""
        patterns = [
            r'ordinance\s*#?\s*(\d+)',
            r'#\s*(\d+)',
            r'number\s*#?\s*(\d+)',
            r'no\.?\s*#?\s*(\d+)',
            r'give\s+me\s+ordinance\s*#?\s*(\d+)',
            r'show\s+me\s+ordinance\s*#?\s*(\d+)',
            r'what\s+is\s+ordinance\s*#?\s*(\d+)',
            r'(?:fines?|penalty|penalties)\s+for\s+(?:ordinance\s*)?#?\s*(\d+)',
            r'when\s+was\s+(?:ordinance\s*)?#?\s*(\d+)',
            r'date\s+.*?(?:ordinance\s*)?#?\s*(\d+)',
        ]

        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        return None

    def _get_ordinance_by_id(self, ordinance_id: str) -> Optional[Dict]:
        """Retrieve a single ordinance by ID if it exists"""
        try:
            ordinance_id = re.sub(r'[^0-9]', '', ordinance_id)
            if not ordinance_id:
                return None

            if ordinance_id not in self.available_ordinance_ids:
                return {
                    "ordinance_id": ordinance_id,
                    "category": "N/A",
                    "short_text": f"Ordinance {ordinance_id} data entry is MISSING from our records.",
                    "fines": "N/A",
                    "date_enacted": "N/A",
                    "status": "N/A",
                    "links": "N/A",
                    "confidence": "Not Found",
                    "score": "0.0%"
                }

            idx = self.id_to_idx.get(ordinance_id)
            if idx is None or idx >= len(self.df):
                return None

            row = self.df.iloc[idx]
            return {
                "ordinance_id": row["ordinance_id"],
                "category": row.get("category", ""),
                "short_text": row["short_text"],
                "fines": row.get("fines", ""),
                "date_enacted": row.get("date_enacted", ""),
                "status": row.get("status", "Status not specified"),
                "links": row.get("links", "Links not specified"),
                "confidence": "High",
                "score": "100.0%"
            }
        except Exception as e:
            logger.warning(f"Error retrieving ordinance by ID {ordinance_id}: {str(e)}")
            return None

    def robust_scale(self, arr: np.ndarray) -> np.ndarray:
        """More robust scaling that handles edge cases"""
        try:
            arr = np.array(arr)
            if np.all(arr == arr[0]):
                return np.ones_like(arr) * 0.5
            return (arr - np.min(arr)) / (np.ptp(arr) + 1e-6)
        except Exception as e:
            logger.warning(f"Error in robust_scale: {str(e)}")
            return np.zeros_like(arr)

    def _get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level"""
        try:
            score = float(score)
            if score > 70:
                return "High"
            elif score > 50:
                return "Medium"
            elif score > 30:
                return "Low"
            return "Very Low"
        except:
            return "Unknown"

    def save_best_checkpoint(self, model_path: str = './best_model'):
        """Save the complete system state"""
        try:
            os.makedirs(model_path, exist_ok=True)

            self.model.save(model_path)

            if self.tokenizer:
                tokenizer_info = {
                    'name_or_path': self.tokenizer.name_or_path,
                    'special_tokens_map': self.tokenizer.special_tokens_map,
                    'init_kwargs': self.tokenizer.init_kwargs
                }
                with open(os.path.join(model_path, 'tokenizer_config.json'), 'w') as f:
                    json.dump(tokenizer_info, f, indent=4)
                self.tokenizer.save_pretrained(model_path)

            if self.df is not None:
                self.df.to_csv(os.path.join(model_path, 'data.csv'), index=False)

            metadata = {
                'best_params': self.best_params,
                'best_checkpoint': getattr(self, 'best_checkpoint', None),
                'retrieval_config': {
                    'default_k': getattr(self, 'default_k', 5),
                    'score_threshold': getattr(self, 'score_threshold', 30.0),
                    'max_same_category': getattr(self, 'max_same_category', 3),
                    'tokenizer_type': 'legal-bert' if 'legal-bert' in str(self.tokenizer) else 'bert-base'
                },
                'components': {
                    'model_type': str(type(self.model)),
                    'cross_encoder_type': str(type(self.cross_encoder)) if self.cross_encoder else None,
                    'bm25_initialized': self.bm25 is not None
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving complete checkpoint: {str(e)}")
            raise

    def load_best_checkpoint(self, model_path: str = './best_model'):
        """Load complete system state"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Checkpoint directory not found at {model_path}")

            metadata_path = os.path.join(model_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                raise FileNotFoundError("Missing metadata file in checkpoint")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_params = metadata.get('best_params', self.best_params)
                self.best_checkpoint = metadata.get('best_checkpoint', None)

                retrieval_config = metadata.get('retrieval_config', {})
                self.default_k = retrieval_config.get('default_k', 5)
                self.score_threshold = retrieval_config.get('score_threshold', 30.0)
                self.max_same_category = retrieval_config.get('max_same_category', 3)

            self.model = SentenceTransformer(model_path)

            tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path) as f:
                    tokenizer_info = json.load(f)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_info['name_or_path'],
                        **tokenizer_info['init_kwargs']
                    )
                except:
                    logger.warning("Failed to load original tokenizer, falling back to default")
                    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            else:
                logger.warning("No tokenizer config found, initializing default tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            data_path = os.path.join(model_path, 'data.csv')
            if os.path.exists(data_path):
                self.load_data(data_path)
            else:
                logger.warning("No data file found in checkpoint")

            if self.df is not None and len(self.df) > 0 and self.tokenizer:
                tokenized_corpus = [self.tokenizer.tokenize(str(text)) for text in self.df["short_text"]]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info("Recreated BM25 index from loaded data")

            if self.df is not None and len(self.df) > 0:
                self.generate_embeddings()
                logger.info("Regenerated embeddings from loaded data")

            if metadata.get('components', {}).get('cross_encoder_type'):
                try:
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                except:
                    logger.warning("Failed to reload cross-encoder")

        except Exception as e:
            logger.error(f"Error loading complete checkpoint: {str(e)}")
            raise

    def train_retrieval_model(self, epochs: int = None, batch_size: int = None,
                            learning_rate: float = None) -> None:
        """Train the retrieval model with contrastive learning"""
        try:
            logger.info("Training retrieval model...")

            epochs = epochs or self.best_params['epochs']
            batch_size = batch_size or self.best_params['batch_size']
            learning_rate = learning_rate or self.best_params['learning_rate']

            batch_size = min(batch_size, 8)

            train_examples = []
            for _, row in self.df.iterrows():
                try:
                    same_cat = self.df[self.df["category"] == row["category"]].sample(1)
                    if len(same_cat) > 0:
                        train_examples.append(InputExample(
                            texts=[str(row["short_text"]), str(same_cat.iloc[0]["short_text"])],
                            label=1.0))

                    diff_cat = self.df[self.df["category"] != row["category"]].sample(1)
                    if len(diff_cat) > 0:
                        train_examples.append(InputExample(
                            texts=[str(row["short_text"]), str(diff_cat.iloc[0]["short_text"])],
                            label=0.0))
                except Exception as e:
                    logger.warning(f"Error creating training example: {str(e)}")
                    continue

            if not train_examples:
                raise ValueError("No valid training examples could be created")

            try:
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
                train_loss = losses.CosineSimilarityLoss(self.model)
            except Exception as e:
                logger.error(f"Error creating dataloader: {str(e)}")
                raise

            warmup_steps = min(100, len(train_dataloader) * epochs // 10)

            try:
                self.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=epochs,
                    warmup_steps=warmup_steps,
                    optimizer_params={'lr': learning_rate},
                    show_progress_bar=True
                )

                self.best_checkpoint = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                self.default_k = 5
                self.score_threshold = 30.0
                self.max_same_category = 3

                self.save_best_checkpoint()

                logger.info("Model training completed successfully")
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in train_retrieval_model: {str(e)}")
            raise

    def process_message(self, user_message: str) -> dict:
        """Process a single message and return formatted response for API endpoints"""
        try:
            # Process the query
            results = self.retrieve_ordinances(user_message)

            response = {
                "message": "",
                "results": []
            }

            # Handle different response types
            if isinstance(results, str):
                # Direct chatbot response (conversational or error)
                response["message"] = results

            elif isinstance(results, tuple):
                message, ordinances = results
                response["message"] = message
                response["results"] = [
                    {
                        "ordinance_id": ord.get("ordinance_id"),
                        "category": ord.get("category"),
                        "short_text": ord.get("short_text"),
                        "fines": ord.get("fines"),
                        "date_enacted": ord.get("date_enacted"),
                        "status": ord.get("status", "Status not specified"),
                        "links": ord.get("links", "Links not specified"),
                        "confidence": ord.get("confidence", "N/A"),
                        "score": ord.get("score", "N/A")
                    }
                    for ord in ordinances[:10]  # Limiting to top 10 results
                ]

            elif isinstance(results, list):
                if not results:
                    response["message"] = "No ordinances found matching your query."
                else:
                    response["message"] = f"Found {len(results)} relevant ordinance(s)"
                    response["results"] = [
                        {
                            "ordinance_id": res.get("ordinance_id"),
                            "category": res.get("category"),
                            "short_text": res.get("short_text"),
                            "fines": res.get("fines"),
                            "date_enacted": res.get("date_enacted"),
                            "status": res.get("status", "Status not specified"),
                            "links": res.get("links", "Links not specified"),
                            "score": res.get("score", "N/A")
                        }
                        for res in results[:10]  # Limiting to top 10 results
                    ]

            # Add disclaimer for ordinance results
            if response["results"]:
                response["disclaimer"] = (
                    "Legal Disclaimer: This is an AI-powered search tool providing general information only. "
                    "For official legal interpretation, please consult with legal professionals or city officials."
                )

            return response

        except Exception as e:
            return {
                "message": f"Error processing request: {str(e)}",
                "results": [],
                "error": True
            }

    def interactive_chatbot_loop(self):
        """Enhanced interactive chatbot loop with RAG capabilities"""
        print("\n" + "="*60)
        print("🤖 JAYOMA BOT - Manila City Ordinance Assistant")
        if self.use_rag:
            print("✨ RAG Mode: Enabled - Using Zephyr language model")
        else:
            print("📋 Classic Mode: Using rule-based responses")
        print("="*60)

        # Display personalized greeting
        print(f"\n{self.get_personalized_greeting()}")
        print("\n💡 Tips:")
        print("• Tell me your name by saying 'My name is [name]'")
        print("• Ask 'what did I search for?' to see your history")
        print("• Search for ordinances by topic or number")
        if self.use_rag:
            print("• Ask complex questions about processes, requirements, or comparisons")
        print("• Type 'help' for more options or 'exit' to quit")
        print("-"*60 + "\n")

        while True:
            try:
                # Get user input
                query = input("You: ").strip()

                if query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print(f"\nJayoma Bot: {self.chatbot.get_farewell()}")
                    # Save memory before exiting
                    self.save_memory()
                    if self.user_name:
                        print(f"See you next time, {self.user_name}!")
                    break

                if not query:
                    print("Jayoma Bot: Please type something so I can help you!")
                    continue

                # Process the query
                start_time = time.time()
                results = self.retrieve_ordinances(query)
                elapsed = time.time() - start_time

                # Handle different response types
                if isinstance(results, str):
                    # Direct chatbot response (conversational or error)
                    print(f"\nJayoma Bot: {results}")
                elif isinstance(results, tuple):
                    # Check if it's a "show all" response
                    if results[0] == "show_all_results":
                        print(f"\nJayoma Bot: Showing all {len(results[1])} results from your previous search:\n")

                        # Display all results
                        for i, res in enumerate(results[1], 1):
                            print(f"📋 #{i} - Ordinance {res['ordinance_id']} ({res['category']})")
                            print(f"   📝 {res['short_text']}")
                            if res['fines']:
                                print(f"   💰 Fines: {res['fines']}")
                            print(f"   📅 Date: {res['date_enacted']} | 🔄 Status: {res['status']}")
                            if res['links'] and res['links'] != "Links not specified":
                                print(f"   🔗 Full Text: {res['links']}")
                            print(f"   📊 Relevance: {res.get('confidence', 'N/A')}")
                            print("-"*50)

                            # Add pagination for readability
                            if i % 5 == 0 and i < len(results[1]):
                                input("\n[Press Enter to continue viewing...]")
                                print()
                    else:
                        # Message with results
                        message, ordinances = results
                        print(f"\nJayoma Bot: {message}")

                        # Check if it's an out-of-scope message
                        if len(ordinances) == 0 and ("couldn't find any ordinances" in message or "outside my area of expertise" in message):
                            continue

                        # Display limited results
                        for i, res in enumerate(ordinances[:10], 1):
                            print(f"\n📋 #{i} - Ordinance {res['ordinance_id']} ({res['category']})")
                            print(f"   📝 {res['short_text'][:150]}...")
                            if res['fines']:
                                print(f"   💰 Fines: {res['fines'][:100]}...")


                        if len(ordinances) > 10:
                            print(f"\n... and {len(ordinances) - 10} more ordinances found.")
                            print("\n💡 Type 'show all' to see all results, or try being more specific!")

                        # Show source ordinances for RAG responses
                        if self.use_rag and len(ordinances) <= 3:
                            print("\n📚 Sources:")
                            for res in ordinances:
                                print(f"   • Ordinance #{res['ordinance_id']} - {res['category']}")

                elif isinstance(results, list):
                    # Regular results list
                    if not results:
                        print(f"\nJayoma Bot: {self.chatbot.get_no_results_message(query)}")
                    else:
                        print(f"\nJayoma Bot: I found {len(results)} relevant ordinance(s):\n")

                        for i, res in enumerate(results, 1):
                            print(f"📋 #{i} - Ordinance {res['ordinance_id']}")
                            print(f"📂 Category: {res['category']} | 📊 Score: {res['score']}")
                            print(f"📝 Summary: {res['short_text']}")
                            if res['fines']:
                                print(f"💰 Fines: {res['fines']}")
                            print(f"📅 Date Enacted: {res['date_enacted']}")
                            print(f"🔄 Status: {res['status']}")
                            if res['links'] and res['links'] != "Links not specified":
                                print(f"🔗 Full Text: {res['links']}")
                            print("-"*50)

                        # Store results and inform about "show all" option if there are many
                        if len(results) > 5:
                            self.last_search_results = results
                            print("\n💡 These are the top results. Type 'show all' to see all matches.")


                # Add disclaimer for ordinance results - ALWAYS show after ordinance details
                if ((isinstance(results, list) and results and results[0].get('ordinance_id', 'N/A') != "N/A") or
                    (isinstance(results, tuple) and len(results) > 1 and results[1] and
                     results[1][0].get('ordinance_id', 'N/A') != 'N/A')):
                    print("\n⚖️ Legal Disclaimer: This is an AI-powered search tool providing general information only.")
                    print("For official legal interpretation, please consult with legal professionals or city officials.")

            except KeyboardInterrupt:
                print("\n\nJayoma Bot: Conversation interrupted. Type 'exit' to quit properly.")
                continue
            except Exception as e:
                print(f"\nJayoma Bot: I encountered an error: {str(e)}")
                print("Please try rephrasing your question or type 'help' for assistance.")
                continue

    def evaluate(self, test_queries: Dict[str, List[str]],
                semantic_weight: float, ce_weight: float) -> float:
        """Evaluate retrieval performance using nDCG - NO LLM DURING EVALUATION"""
        try:
            # CRITICAL: Disable RAG for evaluation
            original_use_rag = self.use_rag
            self.use_rag = False  # Force classic mode for speed

            all_ndcg = []
            for query, relevant_ids in test_queries.items():
                try:
                    results = self.retrieve_ordinances(
                        query,
                        k=5,
                        semantic_weight=semantic_weight,
                        ce_weight=ce_weight
                    )

                    if isinstance(results, str) or isinstance(results, tuple):
                        continue

                    retrieved_ids = [res["ordinance_id"] for res in results]

                    true_relevance = [1 if id in relevant_ids else 0 for id in retrieved_ids]

                    if sum(true_relevance) == 0:
                        continue

                    ideal_relevance = sorted(true_relevance, reverse=True)

                    if len(true_relevance) > 1:
                        ndcg = ndcg_score([true_relevance], [ideal_relevance])
                        all_ndcg.append(ndcg)
                except Exception as e:
                    logger.warning(f"Error evaluating query '{query}': {str(e)}")
                    continue

            # Restore original RAG setting
            self.use_rag = original_use_rag

            return np.mean(all_ndcg) if all_ndcg else 0.0

        except Exception as e:
            logger.error(f"Error in evaluate: {str(e)}")
            return 0.0

    def optimize_hyperparameters(self, test_queries: Dict[str, List[str]],
                               n_trials: int = 20) -> None:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            try:
                params = {
                    'semantic_weight': trial.suggest_float('semantic_weight', 0.4, 0.9),
                    'ce_weight': trial.suggest_float('ce_weight', 0.4, 0.9),
                    'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True),
                    'epochs': trial.suggest_int('epochs', 1, 5)
                }

                self.train_retrieval_model(
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    learning_rate=params['learning_rate']
                )

                self.generate_embeddings()

                ndcg = self.evaluate(
                    test_queries,
                    semantic_weight=params['semantic_weight'],
                    ce_weight=params['ce_weight']
                )

                return ndcg

            except Exception as e:
                logger.error(f"Trial failed: {str(e)}")
                return 0.0

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.best_params.update(study.best_params)
            logger.info(f"Best hyperparameters: {self.best_params}")
            logger.info(f"Best nDCG score: {study.best_value:.4f}")

            self.train_retrieval_model(
                epochs=self.best_params['epochs'],
                batch_size=self.best_params['batch_size'],
                learning_rate=self.best_params['learning_rate']
            )
            self.generate_embeddings()

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise

    def upload_dataset_colab(self) -> str:
        """Handle file upload in Google Colab environment"""
        try:
            from google.colab import files
            uploaded = files.upload()
            if not uploaded:
                raise ValueError("No file was uploaded")
            filename = list(uploaded.keys())[0]
            logger.info(f"Successfully uploaded file: {filename}")
            return filename
        except ImportError:
            logger.error("Google Colab module not found. This function only works in Google Colab environment.")
            raise
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise
