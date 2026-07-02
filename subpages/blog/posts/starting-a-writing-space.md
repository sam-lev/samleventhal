I wanted a low-friction place to write — short notes on things I'm reading, longer pieces when a project is worth explaining, and the occasional photo set. So I added this small blog to the site.

## How it works

There's no database and no build step. Each post is a Markdown file in the `posts/` folder, and a single `posts.json` file lists them in order. The page reads that list, renders whichever post you open, and that's the whole machine.

That keeps everything version-controlled in the repo and means a post can never get "stuck" in some content system — it's just a file.

## Adding a post

1. Write `posts/your-slug.md` in Markdown.
2. Add an entry to the top of `posts.json` with a matching `slug`, a `title`, a `date`, and a short `excerpt`.
3. Commit and push.

Tags are optional; add a `"tags"` array and they become clickable filters automatically.

## Photos

Drop an image into the `media/` folder and reference it from a post like this:

```markdown
![A caption for the photo](media/my-photo.jpg)
```

Paths are relative to the blog page, so `media/my-photo.jpg` is all you need. You can also set a `"cover"` image per post in `posts.json` to give it a banner and a thumbnail in the list.

> One gotcha worth remembering: GitHub Pages is case-sensitive, so `Photo.JPG` and `photo.jpg` are different files.

That's it. More to come.

---

*This is a sample post. Replace it with your own — the file lives at `posts/starting-a-writing-space.md`.*
