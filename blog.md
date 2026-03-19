---
layout: default
title: Blog
---

# Blog

Writing about computer vision, multimodal LLMs, and research notes.

---

{% for post in site.posts %}
### [{{ post.title }}]({{ post.url }})
*{{ post.date | date: "%B %d, %Y" }}*
{% if post.tags and post.tags != empty %} &nbsp;|&nbsp; {% for tag in post.tags %}`{{ tag }}` {% endfor %}{% endif %}

{% if post.description %}{{ post.description }}{% endif %}

---
{% endfor %}

{% if site.posts == empty %}
*No posts yet — check back soon.*
{% endif %}
