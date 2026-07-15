#!/bin/bash
# Stop all healthcare agent services

echo "🛑 Stopping Healthcare Agent Services..."

docker-compose down

echo "✅ Services stopped!"
echo ""
echo "💡 To remove volumes: docker-compose down -v"
echo "💡 To remove images: docker-compose down --rmi all"
