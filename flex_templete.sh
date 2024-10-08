#!/bin/bash

# This script create flex templete by building image using GCP cloud build
# and push the image to artificate registry. Finally, it will create a flex templete for dataflow

# Read project info from config file

CONFIG_FILE="gcp.config"

if [[ ! -f $CONFIG_FILE ]]; then
  echo "Config file not found!"
  exit 1
fi

source $CONFIG_FILE

echo "Project ID: $PROJECT"
echo "Bucket: $BUCKET"
echo "Region: $REGION"
echo "Artifact Registry: $REPOSITORY"


# Ask the user whether to build or pull the image
echo "Do you want to build the image or pull the image? (build/pull/test_local/run_dataflow): " 
read ACTION

case $ACTION in
    build)
        echo "Building the image..."
        # Build image using cloud build
        export TAG=`date +%Y%m%d-%H%M%S`
        
        export SDK_CONTAINER_IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPOSITORY/base_image_dataflow_cityvision:$TAG"

        gcloud builds submit .  --tag $SDK_CONTAINER_IMAGE --project $PROJECT

        export TEMPLATE_FILE=gs://$BUCKET/template_file/dataflow_cityvision-$TAG.json
        export TEMPLATE_IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPOSITORY/dataflow_cityvision_template:$TAG

        echo " Building the flex templete..."

        gcloud dataflow flex-template build $TEMPLATE_FILE  \
                                            --image $SDK_CONTAINER_IMAGE \
                                            --sdk-language "PYTHON" \
                                            --metadata-file=metadata.json \
                                            --project $PROJECT
        ;;

    pull)
        echo "Pulling the image..."
        # Insert the command to pull the image here, for example:
        # docker pull $IMAGE_NAME
        echo "Please enter the image tag: "
        read TAG
        export SDK_CONTAINER_IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPOSITORY/base_image_dataflow_cityvision:$TAG"
        docker pull $SDK_CONTAINER_IMAGE
        ;;

    test_local)

        echo "Testing the image locally..."
        # Insert the command to test the image locally here, for example:
        # docker run $IMAGE_NAME

        export GOOGLE_APPLICATION_CREDENTIALS="application_default_credentials.json"

        python main.py --video_folder dataflow_local_test \
                        --project $PROJECT \
                        --region $REGION \
                        --model_name Yolv8 \
                        --annotation_folder dataflow_local_test/annotation.json \
                        --output_table apps-cityvision-prod.cityvision.test_counting \
                        --runner DirectRunner \
                        --temp_location gs://dataflow-cityvision/dataflow_test_local \
        ;;
    
    run_dataflow)
        
        # GET TAG FROM User
        echo "Please enter the image tag: "
        read TAG
        export TEMPLATE_FILE=gs://$BUCKET/template_file/dataflow_cityvision-$TAG.json
        
        export SDK_CONTAINER_IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPOSITORY/base_image_dataflow_cityvision:$TAG"


        gcloud dataflow flex-template run "flex-`date +%Y%m%d-%H%M%S`" \
            --template-file-gcs-location $TEMPLATE_FILE \
            --project $PROJECT \
            --region "us-west1" \
            --staging-location "gs://dataflow-cityvision/staging" \
            --parameters video_folder="dataflow_prod_test" \
            --parameters annotation_folder="dataflow_prod_test/annotation.json" \
            --parameters model_name="yolv8" \
            --parameters output_table="apps-cityvision-prod.cityvision.test_counting" \
            --parameters sdk_container_image=$SDK_CONTAINER_IMAGE \
            --subnetwork  https://www.googleapis.com/compute/v1/projects/ops-shared-services-hub-prod/regions/us-west1/subnetworks/dsr-dataflow-subnet \
            --disable-public-ips \
            --service-account-email "dsr-dataflow-sa@apps-cityvision-prod.iam.gserviceaccount.com"

        ;;
    *)
        echo "Invalid action!"
        exit 1
        ;;
esac