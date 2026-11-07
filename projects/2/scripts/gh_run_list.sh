#!/usr/bin/env bash

die () {
    msg="$1"

    echo "ERROR: $msg"
    exit 2
}

required_tools() {
    tools="gh jq go"

    for tool in "$tools"; do
        if $(which "$tool") >/dev/null 2>&1; then
            die "$tool: missing!"
        fi
    done
}

REPOSITORY=${REPOSITORY:-""}
WORKFLOW_ID=${WORKFLOW_ID:-""}
LIMIT=${LIMIT:-20}

while [[ $# -gt 0 ]]; do
    case $1 in
        -R|--repository)
        REPOSITORY=$2
        shift
        shift
        ;;
        -w|--workflow)
        WORKFLOW_ID=$2
        shift
        shift
        ;;
        -l|--limit)
        LIMIT=$2
        shift
        shift
        ;;
        *)
        ;;
    esac
done

if [[ -z "$REPOSITORY" ]]; then
    die "Repository is required! Pass with '-R|--repository'"
fi

if [[ -z "$WORKFLOW_ID" ]]; then
    die "Workflow ID is required! Pass with '-w|--workflow'"
fi

GH_TOKEN=$(gh auth token)
if [[ -z "$GH_TOKEN" ]]; then
    die "No GitHub Token available! Run 'gh auth login' and 'gh auth token' right after'."
fi

echo "Repository: $REPOSITORY / Workflow ID: $WORKFLOW_ID / Limit: $LIMIT"
echo

RESULT=$(gh run list -R "$REPOSITORY" -w "$WORKFLOW_ID" --limit "$LIMIT" \
    --json 'number,headBranch,name,createdAt,updatedAt,status,conclusion,attempt' \
    --template '{{ range . }}{{ .name }} - {{ .number }} - {{ .headBranch }} - {{ .createdAt }}/{{ .updatedAt }} - {{ .attempt }} - {{ .status }}/{{ .conclusion}}{{ "\n" }}{{ end }}')

echo "Name - Number - Head Branch - CreatedAt/UpdatedAt - Attempt - Status/Conclusion"
echo "$RESULT"