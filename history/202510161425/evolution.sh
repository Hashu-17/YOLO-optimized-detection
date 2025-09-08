#!/bin/bash

for dir in $(find history -mindepth 1 -maxdepth 1 -type d | sort); do
  echo "Applying snapshot: $dir"

  # clean working directory
  git rm -r --cached . > /dev/null 2>&1
  git clean -fd -e history/ -e evolution.sh -e .gitignore

  # copy snapshot
  cp -r "$dir"/* .

  # format date from folder name
  d=$(basename "$dir")
  formatted_date="${d:0:4}-${d:4:2}-${d:6:2} ${d:8:2}:${d:10:2}:00"

  export GIT_AUTHOR_DATE="$formatted_date"
  export GIT_COMMITTER_DATE="$formatted_date"

  git add .
  if [ -f "$dir/message.txt" ]; then
  	msg=$(cat "$dir/message.txt")	
  else
  	msg="${d#*_}"
  fi

  git commit -m "$msg"	

done
