# Script for creating a packaged repository `repo.zip` to be uploaded to s3 bucket for spark cluster mode deployment.

# This script has dependency on associative arrays and requires bash version 4+.

usage() {
    echo "Usage: $0 -t <path to DAS_2020_DDHCA_Production_Code repository> [-r <path to CEF reader] " 1>&2; exit 1;
}

while getopts t:r: flag
do
    case "$flag" in
        t)
            staging_repo=$OPTARG
            echo "Creating repo.zip from $staging_repo"
            ;;
        r)
            cef_reader_path=$OPTARG
            ;;
        *)
            echo "Invalid argument: $flag"
            usage
            exit 1
            ;;
    esac
done

repo_zip=$(readlink -f $staging_repo)/repo.zip
for m in safetab_p safetab_utils; do
    pushd $staging_repo/$m
    zip -r $repo_zip tmlt
    popd
done

for m in common analytics; do
    pushd $staging_repo/tumult/$m
    zip -r $repo_zip tmlt
    popd
done

pushd $staging_repo
touch "__init__.py"
zip $repo_zip __init__.py
rm "__init__.py"
popd
