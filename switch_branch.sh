#!/bin/bash
# Script to help switch between main and refactor branches

current_branch=$(git branch --show-current)

if [ "$1" == "main" ]; then
    echo "Switching to main branch..."
    # Restore original gitignore when switching to main
    if [ -f ".gitignore.original" ]; then
        cp .gitignore.original .gitignore
    fi
    git checkout main
    echo "✅ Switched to main branch (production)"
    
elif [ "$1" == "refactor" ]; then
    echo "Switching to refactor branch..."
    git checkout refactor/modular-architecture
    # Use refactor gitignore
    if [ -f ".gitignore.original" ]; then
        cp .gitignore .gitignore.refactor 2>/dev/null || true
        cp .gitignore.original .gitignore.main 2>/dev/null || true
    fi
    echo "✅ Switched to refactor/modular-architecture branch"
    echo "⚠️  Remember: This is the TESTING branch"
    
else
    echo "Usage: ./switch_branch.sh [main|refactor]"
    echo "Current branch: $current_branch"
    echo ""
    echo "Options:"
    echo "  main     - Switch to production branch"
    echo "  refactor - Switch to refactoring test branch"
fi