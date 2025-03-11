#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting deployment process...${NC}"

# Check if streamlit URL is provided
if [ -z "$1" ]
then
    echo "Please provide your Streamlit deployment URL"
    echo "Usage: ./deploy.sh YOUR_STREAMLIT_URL"
    exit 1
fi

# Update the Streamlit URL in index.html
echo -e "${BLUE}📝 Updating Streamlit URL in index.html...${NC}"
sed -i "s|YOUR_STREAMLIT_URL|$1|g" index.html

echo -e "${GREEN}✅ Deployment files prepared successfully!${NC}"
echo -e "${BLUE}📋 Next steps:${NC}"
echo "1. Commit these changes to your repository"
echo "2. Go to Netlify and connect your repository"
echo "3. Set the following deployment settings:"
echo "   - Base directory: netlify"
echo "   - Build command: leave empty"
echo "   - Publish directory: netlify"
echo -e "${GREEN}🎉 You're ready to deploy!${NC}" 