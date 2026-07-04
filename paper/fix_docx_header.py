#!/usr/bin/env python3
"""Post-render fix: inject the APA running head + page number into the docx headers.

apaquarto's docx output carries a page-number field in the default header but no
running-head text, and leaves the first/even headers empty. This script rewrites all
header parts with the running head (upper-case short title), a right tab, and a PAGE
field, so every page shows "SHORT TITLE <tab> N" as APA requires.

Usage: python3 fix_docx_header.py [anchor_dif.docx]
"""
import re
import shutil
import sys
import zipfile

DOCX = sys.argv[1] if len(sys.argv) > 1 else "anchor_dif.docx"
RUNNING_HEAD = "CONTAMINATION RARELY REORDERS LEADERBOARDS"

BODY = (
    "<w:p><w:pPr><w:pStyle w:val=\"Header\"/>"
    "<w:tabs><w:tab w:val=\"right\" w:pos=\"9350\"/></w:tabs></w:pPr>"
    f"<w:r><w:t xml:space=\"preserve\">{RUNNING_HEAD}</w:t></w:r>"
    "<w:r><w:tab/></w:r>"
    "<w:r><w:fldChar w:fldCharType=\"begin\"/></w:r>"
    "<w:r><w:instrText xml:space=\"preserve\"> PAGE \\* MERGEFORMAT </w:instrText></w:r>"
    "<w:r><w:fldChar w:fldCharType=\"separate\"/></w:r>"
    "<w:r><w:t>1</w:t></w:r>"
    "<w:r><w:fldChar w:fldCharType=\"end\"/></w:r>"
    "</w:p>"
)

tmp = DOCX + ".tmp"
with zipfile.ZipFile(DOCX) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    n_fixed = 0
    for item in zin.infolist():
        data = zin.read(item.filename)
        if re.fullmatch(r"word/header\d+\.xml", item.filename):
            xml = data.decode("utf-8")
            m = re.match(r"(.*?<w:hdr[^>]*>).*(</w:hdr>)", xml, re.S)
            if m:
                data = (m.group(1) + BODY + m.group(2)).encode("utf-8")
                n_fixed += 1
        zout.writestr(item, data)
shutil.move(tmp, DOCX)
print(f"fixed {n_fixed} header part(s) in {DOCX}: running head + page number")
