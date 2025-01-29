---
layout: post
title:  "Deepfood - Image Retrieval System in Production (test)"
excerpt: "Build and deploy a food image retrieval system with pytorch, annoy, starlette and AWS Elastic Beanstalk"
date:   2020-05-15 06:48:38 +0200
hide: false
categories: deep-learning transfer-learning annoy ann aws
permalink: /2020/05/15/deep-search-aws.html/
---

In this post I will build and **deploy** a food image retrieval system. I will use pytorch to train a model that extracts image features, Annoy for finding nearest neighbor images for a given query image, starlette for building a web application and AWS Elastic Beanstalk for deploying the app. Lets begin!  
The full code is here: [github](https://github.com/yonigottesman/deepfood).  

Introduction
==
The goal of an image retrieval system is to let a user send a query image and return a list of most similar images to the query. For example with [google reverse image search](https://support.google.com/websearch/answer/1325808?co=GENIE.Platform%3DDesktop&hl=en) you can send an image and get all similar images and their web pages. Amazon also has an option to [search by image](https://www.amazon.com/b?ie=UTF8&node=17387598011), just take a picture of something you see and immediately get a list of items sold by amazon that look the same as your picture.  

I'm going to build an application to retrieve **food** images and deploy it as a web service.
