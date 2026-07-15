#!/bin/bash
# Build and start all healthcare agent services

echo "🏗️  Building Healthcare Agent Services..."

# Build all images
docker-compose build --parallel

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting services..."
    docker-compose up -d
    
    echo ""
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
    
    echo ""
    echo "📊 Service Status:"
    docker-compose ps
    
    echo ""
    echo "✅ Healthcare Agent System is running!"
    echo ""
    echo "🌐 API Endpoints:"
    echo "   Supervisor API:  http://localhost:8000"
    echo "   API Docs:        http://localhost:8000/docs"
    echo "   PubMed Agent:    http://localhost:8001"
    echo "   WebSearch Agent: http://localhost:8002"
    echo "   Synthesis Agent: http://localhost:8003"
    echo ""
    echo "📝 Test the API:"
    echo '   curl -X POST http://localhost:8000/query \\'
    echo '        -H "Content-Type: application/json" \\'
    echo '        -d '"'"'{"query": "What are the symptoms of diabetes?"}'"'"
    echo ""
    echo "🛑 To stop: docker-compose down"
    echo "📋 To view logs: docker-compose logs -f"
else
    echo "❌ Build failed!"
    exit 1
fi
