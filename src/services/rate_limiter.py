"""Rate limiting service for API calls."""
import time
from typing import Dict, List, Optional, Tuple


class RateLimiter:
    """Rate limiter for controlling API usage."""
    
    def __init__(self, max_calls: int, interval: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed per interval
            interval: Time interval in seconds
        """
        self.max_calls = max_calls
        self.interval = interval
        self.calls: Dict[int, List[float]] = {}
        
    def is_rate_limited(self, user_id: int) -> bool:
        """
        Check if a user is rate limited.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            True if rate limited, False otherwise
        """
        current_time = time.time()
        
        # Initialize user's call list if not exists
        self.calls.setdefault(user_id, [])
        
        # Remove calls outside the interval window
        self.calls[user_id] = [
            t for t in self.calls[user_id] 
            if current_time - t <= self.interval
        ]
        
        # Check if rate limit exceeded
        if len(self.calls[user_id]) >= self.max_calls:
            return True
            
        # Record this call
        self.calls[user_id].append(current_time)
        return False
        
    def check_rate_limit(self, user_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if an API call can be made.
        
        Args:
            user_id: Discord user ID (optional)
            
        Returns:
            Tuple of (can_make_call, error_message)
        """
        # If no user_id provided, allow the call (system calls)
        if user_id is None:
            return True, None
            
        if self.is_rate_limited(user_id):
            time_until_reset = self.get_time_until_reset(user_id)
            error_msg = (
                f"🚫 Rate limit exceeded. Please wait {time_until_reset} seconds "
                f"before making another request."
            )
            return False, error_msg
            
        return True, None
        
    def get_time_until_reset(self, user_id: int) -> int:
        """
        Get time in seconds until rate limit resets for a user.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Seconds until rate limit resets
        """
        if user_id not in self.calls or not self.calls[user_id]:
            return 0
            
        oldest_call = min(self.calls[user_id])
        current_time = time.time()
        time_passed = current_time - oldest_call
        
        if time_passed >= self.interval:
            return 0
            
        return int(self.interval - time_passed)
        
    def get_remaining_calls(self, user_id: int) -> int:
        """
        Get remaining API calls for a user.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Number of remaining calls
        """
        current_time = time.time()
        
        # Clean up old calls
        if user_id in self.calls:
            self.calls[user_id] = [
                t for t in self.calls[user_id] 
                if current_time - t <= self.interval
            ]
            return max(0, self.max_calls - len(self.calls[user_id]))
            
        return self.max_calls
        
    def reset_user(self, user_id: int) -> None:
        """
        Reset rate limit for a specific user.
        
        Args:
            user_id: Discord user ID
        """
        if user_id in self.calls:
            del self.calls[user_id]
            
    def reset_all(self) -> None:
        """Reset rate limits for all users."""
        self.calls.clear()