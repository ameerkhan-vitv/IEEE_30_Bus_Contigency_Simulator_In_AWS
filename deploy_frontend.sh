#!/usr/bin/env bash
# Fills the API URL into index.html and uploads it to the site bucket.
#   bash deploy_frontend.sh            (stack grid-sim, region ap-south-1)
set -euo pipefail
STACK=${1:-grid-sim}; REGION=${2:-ap-south-1}
out() { aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
API=$(out ApiUrl); BUCKET=$(out SiteBucketName); SITE=$(out SiteUrl)
sed -e "s|__API__|$API|g" \
    frontend/index.html > /tmp/index.html
aws s3 cp /tmp/index.html "s3://$BUCKET/index.html" \
    --content-type "text/html; charset=utf-8" --cache-control "no-cache" --region "$REGION"
echo "Done. Open: $SITE"
