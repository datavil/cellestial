VERSION=$(poetry version -s)

gh release create "v${VERSION}" \
  --repo datavil/cellestial \
  --target master \
  --title "Cellestial ${VERSION}" \
  --notes-file release-notes.md