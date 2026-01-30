#!/bin/bash

# Import GraphRAG GraphML into Neo4j
# Usage: ./import_graphml.sh

set -e

echo "🚀 GraphRAG GraphML Import Script"
echo "=================================="
echo ""

# Configuration
NEO4J_USER="neo4j"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-speedkg123}"
GRAPHML_PATH="/import/graph.graphml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if Neo4j is running
echo "📋 Step 1: Checking Neo4j status..."
if ! docker-compose ps | grep -q "neo4j.*Up"; then
    echo -e "${YELLOW}⚠️  Neo4j is not running. Starting Neo4j...${NC}"
    docker-compose up -d
    echo "⏳ Waiting 30 seconds for Neo4j to start..."
    sleep 30
else
    echo -e "${GREEN}✅ Neo4j is running${NC}"
fi

# Step 2: Wait for Neo4j to be ready
echo ""
echo "📋 Step 2: Waiting for Neo4j to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker-compose exec -T neo4j cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD "RETURN 1;" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Neo4j is ready${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "⏳ Waiting for Neo4j... (attempt $ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}❌ Neo4j failed to start after $MAX_ATTEMPTS attempts${NC}"
    exit 1
fi

# Step 3: Verify GraphML file exists
echo ""
echo "📋 Step 3: Verifying GraphML file..."
if docker-compose exec -T neo4j test -f $GRAPHML_PATH; then
    FILE_SIZE=$(docker-compose exec -T neo4j stat -f%z $GRAPHML_PATH 2>/dev/null || docker-compose exec -T neo4j stat -c%s $GRAPHML_PATH 2>/dev/null)
    echo -e "${GREEN}✅ GraphML file found: $GRAPHML_PATH ($(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes"))${NC}"
else
    echo -e "${RED}❌ GraphML file not found at $GRAPHML_PATH${NC}"
    echo ""
    echo "Please ensure:"
    echo "1. GraphRAG indexing has been run"
    echo "2. Snapshots are enabled in settings.yaml (snapshots.graphml: true)"
    echo "3. The output directory path is correct in docker-compose.yml"
    exit 1
fi

# Step 4: Import GraphML
echo ""
echo "📋 Step 4: Importing GraphML into Neo4j..."
echo "This may take a few minutes depending on graph size..."
echo ""

docker-compose exec -T neo4j cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD <<EOF
CALL apoc.import.graphml(
  '$GRAPHML_PATH',
  {
    batchSize: 1000,
    compression: 'NONE',
    source: {
      label: 'Entity'
    },
    target: {
      label: 'Entity'
    }
  }
)
YIELD file, source, format, nodes, relationships, properties, time
RETURN file, nodes, relationships, properties, time;
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ GraphML import completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ GraphML import failed${NC}"
    exit 1
fi

# Step 5: Verify import
echo ""
echo "📋 Step 5: Verifying import..."
echo ""

NODE_COUNT=$(docker-compose exec -T neo4j cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD --format plain "MATCH (n) RETURN count(n);" | tail -1)
REL_COUNT=$(docker-compose exec -T neo4j cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD --format plain "MATCH ()-[r]->() RETURN count(r);" | tail -1)

echo -e "${GREEN}✅ Import verification:${NC}"
echo "   Nodes: $NODE_COUNT"
echo "   Relationships: $REL_COUNT"

# Step 6: Show sample data
echo ""
echo "📋 Step 6: Sample entities (top 10 by degree)..."
echo ""

docker-compose exec -T neo4j cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD <<EOF
MATCH (n:Entity)
WHERE n.degree IS NOT NULL
RETURN n.label as entity, n.type as type, n.degree as connections
ORDER BY n.degree DESC
LIMIT 10;
EOF

echo ""
echo -e "${GREEN}🎉 Import complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Open Neo4j Browser: http://localhost:7474"
echo "2. Login with neo4j / $NEO4J_PASSWORD"
echo "3. Try queries from NEO4J_SETUP.md"
echo ""
echo "Example query to get started:"
echo ""
echo "  MATCH (n:Entity)-[r]-(m:Entity)"
echo "  WHERE n.degree > 5"
echo "  RETURN n, r, m"
echo "  LIMIT 50;"
echo ""
