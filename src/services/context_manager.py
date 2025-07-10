"""Context management for user conversations."""
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ..config.settings import get_settings


class ContextManager:
    """Manages conversation context for users."""
    
    _instance: Optional['ContextManager'] = None
    
    def __init__(self):
        """Initialize context manager."""
        self.settings = get_settings()
        self.user_contexts: Dict[int, Deque[Dict[str, Any]]] = {}
        
    @classmethod
    def get_instance(cls) -> 'ContextManager':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_user_context(self, user_id: int) -> Deque[Dict[str, Any]]:
        """
        Get context for a user, creating if necessary.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            User's context deque
        """
        return self.user_contexts.setdefault(
            user_id, 
            deque(maxlen=self.settings.max_context_messages)
        )
        
    def update_context(self, user_id: int, content: str, role: str) -> None:
        """
        Update user's conversation context.
        
        Args:
            user_id: Discord user ID
            content: Message content
            role: Message role (user/assistant/system)
        """
        context = self.get_user_context(user_id)
        current_time = time.time()
        
        # Initialize with system prompt if empty
        if not context and role == 'user':
            context.append({
                'role': 'system',
                'content': self.settings.system_prompt.strip(),
                'timestamp': current_time,
            })
            
        # Validate role alternation
        if context:
            last_role = context[-1]['role']
            
            # Check for valid role transitions
            valid_transition = (
                (last_role == 'system' and role == 'user') or
                (last_role == 'user' and role == 'assistant') or
                (last_role == 'assistant' and role == 'user')
            )
            
            if not valid_transition:
                return  # Skip invalid transitions
                
        # Add message to context
        context.append({
            'role': role,
            'content': content.strip(),
            'timestamp': current_time,
        })
        
        # Clean up old messages
        self._cleanup_old_messages(user_id)
        
    def get_context_messages(self, user_id: int) -> List[Dict[str, str]]:
        """
        Get formatted context messages for API calls.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            List of formatted messages
        """
        context = self.get_user_context(user_id)
        
        # Convert to API format
        messages = [
            {"role": msg['role'], "content": msg['content']} 
            for msg in context
        ]
        
        # Ensure system message is first
        if not messages or messages[0]['role'] != 'system':
            messages.insert(0, {
                "role": "system",
                "content": self.settings.system_prompt.strip(),
            })
            
        return self._validate_message_order(messages)
        
    def _cleanup_old_messages(self, user_id: int) -> None:
        """Remove messages older than max context age."""
        if user_id not in self.user_contexts:
            return
            
        current_time = time.time()
        cutoff_time = current_time - self.settings.max_context_age
        
        # Filter out old messages
        context = self.user_contexts[user_id]
        self.user_contexts[user_id] = deque(
            [msg for msg in context if msg['timestamp'] >= cutoff_time],
            maxlen=self.settings.max_context_messages
        )
        
    def _validate_message_order(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure messages alternate roles correctly."""
        if not messages:
            return messages
            
        cleaned = [messages[0]]  # Start with system message
        
        for i in range(1, len(messages)):
            last_role = cleaned[-1]['role']
            current_role = messages[i]['role']
            
            # Determine expected role
            if last_role in ['system', 'assistant']:
                expected_role = 'user'
            elif last_role == 'user':
                expected_role = 'assistant'
            else:
                continue  # Skip unknown roles
                
            if current_role == expected_role:
                cleaned.append(messages[i])
                
        return cleaned
        
    def clear_context(self, user_id: int) -> None:
        """Clear context for a specific user."""
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]
            
    def has_context(self, user_id: int) -> bool:
        """Check if user has any context."""
        return user_id in self.user_contexts and len(self.user_contexts[user_id]) > 0
        
    def get_context_summary(self, user_id: int) -> Dict[str, Any]:
        """Get summary information about user's context."""
        if user_id not in self.user_contexts:
            return {"messages": 0, "oldest_timestamp": None}
            
        context = self.user_contexts[user_id]
        if not context:
            return {"messages": 0, "oldest_timestamp": None}
            
        return {
            "messages": len(context),
            "oldest_timestamp": context[0]['timestamp'],
            "newest_timestamp": context[-1]['timestamp'],
            "roles": [msg['role'] for msg in context]
        }