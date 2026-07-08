# Writing (blog)

A self-contained blog for the site. No build step, no database — each post is a
Markdown file, and `posts.json` lists them. Everything is version-controlled in the repo.

## Structure

```
subpages/blog/
  index.html        the blog (hash-routed: list, post, tag views)
  posts.json        ordered list of posts (newest first)
  posts/            one .md file per post (body only)
  media/            images and cover art
  lib/marked.min.js vendored Markdown renderer (marked v12, MIT) — don't edit
```

## Add a post

1. Write the body in `posts/your-slug.md` (plain Markdown).
2. Add an entry to the **top** of `posts.json`:

   ```json
   {
     "slug": "your-slug",
     "title": "Your title",
     "date": "2026-06-20",
     "excerpt": "One sentence shown in the list.",
     "tags": ["topology", "notes"],
     "cover": "media/your-cover.jpg"
   }
   ```

   - `slug` must match the filename (without `.md`).
   - `date` is `YYYY-MM-DD`.
   - `tags` and `cover` are optional. Tags become clickable filters automatically.
   - Add `"draft": true` to keep a post in the repo but hidden from the site.
3. Commit and push.

The post title comes from `posts.json`, so you don't need to repeat it as a heading
in the `.md` file — just start writing the body.

## Photos

Put the image in `media/`, then reference it in a post:

```markdown
![Alt text](media/my-photo.jpg)
```

Paths are relative to the blog page, so `media/filename` is correct (no leading slash).
GitHub Pages is case-sensitive: `Photo.JPG` ≠ `photo.jpg`.

## Local preview

Because the page fetches `.md` files, open it over HTTP, not by double-clicking:

```bash
# from the repo root
python3 -m http.server
# then visit http://localhost:8000/subpages/blog/
```

On the live GitHub Pages site no server is needed — it just works.

## Notes

- The nav links in `index.html` point back to the main site (`../../`,
  `../../#research`, `../../#projects`). Confirm those section ids exist on the
  homepage.
- Rename the blog title/tagline in the `BLOG` config object near the top of the
  `<script>` in `index.html`.
