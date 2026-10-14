#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data QA — the three hard rules: zero HTML tags in the thinking region / <think> closed exactly
once / the ai-view card opens and closes completely.
Usage: python check_data.py [data/lima_html_830_clean.json]"""
import json, re, sys
from html.parser import HTMLParser


class TagCounter(HTMLParser):
    """Count the total number of tags (opening + closing) in a piece of text.
    More reliable than count('<'): text like `a < b` will not be misread as a tag."""

    def __init__(self):
        super().__init__()
        self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1

    def handle_endtag(self, tag):
        self.tags += 1
    # Self-closing and void tags are counted too (handle_startendtag goes through starttag + endtag)


def html_tags(text: str) -> int:
    p = TagCounter()
    p.feed(text)
    return p.tags


# Whitelist for pseudo-labels from a code domain: for these samples the "tags" are the programming
# syntax of the question itself (e.g. <xs:element> in an XML Schema lesson, the Java generic List<T>,
# a git <branch> placeholder). Deleting them would destroy the question's meaning — see README §5,
# "pseudo-labels vs real leakage".
CODE_DOMAIN_WHITELIST = {
    "lima_0027",  # <xs:element> in an XML Schema question
    "lima_0082",  # git <branch>
    "lima_0097",  # C# generics <T> / <K,V>
    "lima_0123",  # git <tagname>
    "lima_0124",  # git submodule <url> <path>
    "lima_0190",  # C++ std::function<void()>
    "lima_0197",  # TypeScript <T>
    "lima_0213",  # Java Map<String, Integer>
    "lima_0223",  # TS Promise<number>[]
    "lima_0344",  # C++ <windows.h>
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/lima_html_830_clean.json"
    data = json.load(open(path, encoding="utf-8"))
    bad, stats = [], {"cot": len([d for d in data if d.get("has_cot")]),
                      "direct": len(data) - len([d for d in data if d.get("has_cot")])}
    print(f"Total {len(data)} samples (CoT {stats['cot']} + direct {stats['direct']})")

    for d in data:
        if d["id"] in CODE_DOMAIN_WHITELIST:
            continue
        gpt = d["conversations"][1]["value"]
        if d.get("has_cot"):
            m = re.match(r"^<think>\n?(.*?)</think>", gpt, re.S)
            if not m:
                bad.append((d["id"], "malformed <think> open/close"))
                continue
            n1, n2 = html_tags(d.get("cot", "")), html_tags(m.group(1))
            if n1 > 0:
                bad.append((d["id"], f"cot field contains {n1} tags"))
            if n2 > 0:
                bad.append((d["id"], f"think body contains {n2} tags"))
            body = gpt[m.end():]
        else:
            body = gpt
        if not body.lstrip().startswith('<div class="ai-view">'):
            bad.append((d["id"], "body does not start with ai-view"))
        elif not body.rstrip().endswith("</div>"):
            bad.append((d["id"], "body does not end with </div>"))

    if bad:
        print(f"❌ {len(bad)} dirty samples:")
        for i, x in enumerate(bad[:15]):
            print("  ", x)
        if len(bad) > 15:
            print(f"   ... {len(bad)} in total")
        sys.exit(1)
    print("✅ Dataset clean: zero HTML tags in the thinking region, all cards open and close properly")


if __name__ == "__main__":
    main()
