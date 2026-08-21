param(
  [string]$Root = (Get-Location).Path
)

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$DocsRoot = Join-Path $Root 'docs'
$WorkflowRoot = Join-Path $Root '.github\workflows'

function Write-Utf8File {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Content
  )

  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }

  [System.IO.File]::WriteAllText($Path, $Content, $Utf8NoBom)
}

function YamlQuote {
  param([Parameter(Mandatory = $true)][string]$Value)
  '"' + ($Value -replace '"', '\"') + '"'
}

function Get-DisplayTitle {
  param([Parameter(Mandatory = $true)][string]$BaseName)

  $title = $BaseName -replace '^\d+_[^-]+-', ''
  switch ($title) {
    '残差网络（ResNet）' { 'ResNet' }
    default { $title }
  }
}

function Clean-Markdown {
  param([Parameter(Mandatory = $true)][string]$Text)

  $pairs = @(
    @('\_', '_'),
    @('\-', '-'),
    @('\.', '.'),
    @('\(', '('),
    @('\)', ')'),
    @('\~', '~'),
    @('本文和大家聊聊关于', '这一页整理'),
    @('本小节咱们来聊聊', '这一节聚焦'),
    @('咱们', '我们'),
    @('简单来说，', '先用一句话抓住它：'),
    @('简单来说', '先用一句话抓住它'),
    @('总的来说，', '整体上，'),
    @('总的来说', '整体上'),
    @('举个例子', '例如'),
    @('比如', '例如'),
    @('下面我来', '接下来整理'),
    @('下面我来详细介绍', '接下来详细说明'),
    @('接下来我们来', '接着看'),
    @('为了方便理解', '便于理解'),
    @('核心思想', '关键思路'),
    @('最常见的是', '常见的是'),
    @('最终', '最后'),
    @('详细学习', '展开学习'),
    @('详细介绍', '说明'),
    @('详细讲解', '说明'),
    @('详细理解', '理解'),
    @('常常', '经常'),
    @('非常非常', '非常'),
    @('能够', '可以'),
    @('让模型', '让模型更容易'),
    @('原理', '机制'),
    @('案例', '例子'),
    @('知识点', '要点')
  )

  foreach ($pair in $pairs) {
    $Text = $Text.Replace($pair[0], $pair[1])
  }

  $Text = $Text -replace "(\r?\n){3,}", "`r`n`r`n"
  return $Text.Trim()
}

function Build-Article {
  param(
    [Parameter(Mandatory = $true)][string]$Category,
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$Body
  )

  $categoryMeta = @{
    '参数估计方法' = @{
      intro = '这一页把 {title} 放回“从数据反推参数”的主线里，保留公式，但把叙述改得更适合公开分享。'
      bullets = @(
        '先明确概率模型和未知参数。',
        '再选择似然、误差或后验作为目标函数。',
        '最后看结果是否便于计算、解释和落地。'
      )
    }
    '异常值处理方法' = @{
      intro = '这一页重新梳理 {title} 的判断规则和处理思路，尽量让读者先看懂“为什么”，再看“怎么做”。'
      bullets = @(
        '先判断点位是不是明显偏离主体分布。',
        '再决定是删除、替换、截断还是分组处理。',
        '处理后最好再检查分布是否更稳。'
      )
    }
    '神经网络' = @{
      intro = '这一页把 {title} 的结构、直觉和训练要点重新组织了一遍，便于查阅和分享。'
      bullets = @(
        '先抓住网络由什么模块组成。',
        '再看它解决了什么类型的问题。',
        '最后记住训练时最容易出问题的地方。'
      )
    }
  }

  $meta = $categoryMeta[$Category]
  $preface = @"
---
title: $(YamlQuote $Title)
description: $(YamlQuote "$Category 主题下的 $Title 条目，已重写整理为适合公开分享的版本。")
---

> $($meta.intro.Replace('{title}', $Title))

## 快速理解
$(($meta.bullets | ForEach-Object { "- $_" }) -join "`r`n")
"@

  return ($preface.Trim() + "`r`n`r`n" + $Body.Trim())
}

function Build-IndexPage {
  param(
    [Parameter(Mandatory = $true)][string]$Category,
    [Parameter(Mandatory = $true)][object[]]$Entries
  )

  $title = $Category
  $body = @"
# $title

这一栏收录了 $title 主题的重写版笔记，保留核心公式与思路，读起来更像一套公开知识库。

## 页面
$(($Entries | ForEach-Object { "- [$($_.Display)]($($_.File))" }) -join "`r`n")
"@

  return $body.Trim()
}

function Build-HomePage {
  return @"
# 机器学习知识库

这是把原始笔记重写整理后的公开版本，按主题分成三个板块，适合直接发到 GitHub Pages。

## 目录

- [参数估计方法](参数估计方法/index.md)
- [异常值处理方法](异常值处理方法/index.md)
- [神经网络](神经网络/index.md)

## 说明

页面保留了原来的知识点，但统一了标题、结构和表达方式，方便检索和分享。
"@
}

function Build-Nav {
  param([Parameter(Mandatory = $true)][hashtable]$CategoryPages)

  $order = @('参数估计方法', '异常值处理方法', '神经网络')
  $lines = @('nav:', '  - 首页: index.md')

  foreach ($category in $order) {
    if (-not $CategoryPages.ContainsKey($category)) {
      continue
    }

    $lines += "  - $($category):"
    $lines += "      - 概览: $category/index.md"

    foreach ($entry in ($CategoryPages[$category] | Sort-Object Display)) {
      $lines += "      - $($entry.Display): $category/$($entry.File)"
    }
  }

  return $lines -join "`r`n"
}

function Build-MkDocsYml {
  param([Parameter(Mandatory = $true)][string]$NavText)

  return @"
site_name: 机器学习知识库
site_description: 面向知识分享的机器学习笔记站
theme:
  name: material
  language: zh
  features:
    - navigation.instant
    - navigation.sections
    - navigation.tracking
    - navigation.top
    - search.highlight
    - search.share
    - content.code.copy
plugins:
  - search
markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - toc:
      permalink: true
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.details
  - pymdownx.superfences
extra_javascript:
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
$NavText
"@
}

function Build-Requirements {
  return @"
mkdocs-material
pymdown-extensions
"@
}

function Build-Workflow {
  return @'
name: Deploy MkDocs site

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build site
        run: mkdocs build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        name: Deploy
        uses: actions/deploy-pages@v4
'@
}

function Build-Readme {
  return @"
# 机器学习知识库

这是一个用 MkDocs Material 整理出来的机器学习笔记站，源内容已经被重写和重排，适合直接发布到 GitHub Pages。

## 本地预览

1. 安装 Python 3.11+。
2. 运行 `pip install -r requirements.txt`。
3. 运行 `mkdocs serve`。

## 发布到 GitHub Pages

1. 把仓库推到 GitHub。
2. 在仓库设置里把 Pages 来源切到 GitHub Actions。
3. 推送到 `main` 后，工作流会自动构建并发布。
"@
}

if (-not (Test-Path -LiteralPath $DocsRoot)) {
  New-Item -ItemType Directory -Path $DocsRoot -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $WorkflowRoot)) {
  New-Item -ItemType Directory -Path $WorkflowRoot -Force | Out-Null
}

Write-Utf8File -Path (Join-Path $Root 'requirements.txt') -Content (Build-Requirements)
Write-Utf8File -Path (Join-Path $Root 'README.md') -Content (Build-Readme)
Write-Utf8File -Path (Join-Path $WorkflowRoot 'pages.yml') -Content (Build-Workflow)

$categoryPages = @{}

$markdownFiles = Get-ChildItem -Path $Root -Recurse -File -Filter *.md | Where-Object {
  $_.FullName -notlike "$DocsRoot*" -and
  $_.FullName -notlike (Join-Path $Root '.github*') -and
  $_.FullName -notlike (Join-Path $Root 'tools*')
}

foreach ($file in $markdownFiles) {
  $rel = $file.FullName.Substring($Root.Length + 1)
  if ($rel -eq '欢迎.md') {
    Write-Utf8File -Path (Join-Path $DocsRoot 'index.md') -Content (Build-HomePage)
    continue
  }

  $category = Split-Path -Parent $rel
  if ([string]::IsNullOrWhiteSpace($category)) {
    continue
  }

  if (-not $categoryPages.ContainsKey($category)) {
    $categoryPages[$category] = @()
  }

  $base = [System.IO.Path]::GetFileNameWithoutExtension($rel)
  $display = Get-DisplayTitle -BaseName $base
  $raw = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
  $lines = $raw -split "`r?`n"
  $sectionStart = [Array]::FindIndex($lines, [Predicate[string]]{ param($line) $line -match '^##\s' })
  if ($sectionStart -ge 0) {
    $body = ($lines[$sectionStart..($lines.Count - 1)] -join "`r`n")
  } else {
    $body = ($lines[1..($lines.Count - 1)] -join "`r`n")
  }

  $body = Clean-Markdown -Text $body
  $article = Build-Article -Category $category -Title $display -Body $body

  $target = Join-Path $DocsRoot $rel
  Write-Utf8File -Path $target -Content $article

  $categoryPages[$category] += [pscustomobject]@{
    Display = $display
    File = ([System.IO.Path]::GetFileName($rel))
  }
}

foreach ($category in $categoryPages.Keys) {
  $entries = $categoryPages[$category] | Sort-Object Display
  $indexContent = Build-IndexPage -Category $category -Entries $entries
  Write-Utf8File -Path (Join-Path $DocsRoot (Join-Path $category 'index.md')) -Content $indexContent
}

Write-Utf8File -Path (Join-Path $Root 'mkdocs.yml') -Content (Build-MkDocsYml -NavText (Build-Nav -CategoryPages $categoryPages))

$imageFiles = Get-ChildItem -Path $Root -Recurse -File -Include *.png, *.jpg, *.jpeg, *.gif, *.webp | Where-Object {
  $_.FullName -notlike "$DocsRoot*"
}

foreach ($image in $imageFiles) {
  $rel = $image.FullName.Substring($Root.Length + 1)
  $dest = Join-Path $DocsRoot (Join-Path 'assets' $rel)
  $destDir = Split-Path -Parent $dest
  if (-not (Test-Path -LiteralPath $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
  }
  Copy-Item -LiteralPath $image.FullName -Destination $dest -Force
}

Write-Host "Generated MkDocs source under $DocsRoot"
