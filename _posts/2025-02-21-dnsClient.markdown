---
layout: post
title: "Understanding DNS: Building a Client"
date: 2025-02-21 14:30:00 +0000
description: "Building a simple DNS client in Python to understand the internals of DNS queries and responses, and the broader internet process."
img: ../img/dns/dns.png
tags: [DNS, Networking, Education, Python]
---

## Introduction

As part of my coursework in Computer Networking, I extended my learning my embarking on a challenge to build a DNS client in Python to gain some valueable experience trying to encode, send, receive, and decode DNS queries in a manner similar to the program `nslookup`. This was inspired by the fact that I worked with many different protocols and RFC and SMPTE protocols in my professional internship, but am not able to discuss the applications of them. I would like this to be an example of my ability to read technical protocol documentation and devlelop software ontop of it.

This project serves as a learning tool to explore the intricacies of DNS based on RFC 1035. It highlights the process of building a query, parsing the response, and understanding how DNS records (like A and CNAME) are encoded. T

---

## Purpose

The primary purpose of this project is educational:
- **Build Applications Atop Protocols:** Understand how to interpret techincal documentation and use it to build useful software.
- **Research & Exploration:** Delving into the underlying process of how DNS queries and responses are structured.
- **Understanding the Internet:** Gaining insight into how domain names translate into IP addresses and how information can be queried from DNS servers.
- **Practical Coding Experience:** Implementing networking code in Python to interact with real-world DNS servers.

---

## Technical Overview

### How It Works

1. **DNS Query Construction:**  
   The client constructs a DNS query in compliance with RFC 1035. This involves setting up header fields, encoding a domain name into DNS label format, and appending query type and class.

2. **Sending and Receiving:**  
   The query is sent over UDP to a specified DNS server. The response is received as raw bytes and parsed to extract key sections, such as the header and answer records.

3. **Response Parsing:**  
   Special handling is implemented for A records (IPv4 addresses) and CNAME records (canonical names). Compressed domain names are decoded using pointer reference techniques described in RFC 1035.

4. **Pretty Printing:**  
   For clarity and ease of understanding, a custom pretty-print function formats the DNS response in a user-friendly way, emulating the output of the standard `nslookup` utility.

---

## Code

[Github Link](https://github.com/mcassar4/dnsClient). 
<pre><code id="code-block"></code></pre>

<script>
fetch('https://raw.githubusercontent.com/mcassar4/dnsClient/main/dnsClient.py')
  .then(response => response.text())
  .then(text => {
    document.getElementById('code-block').textContent = text;
  });
</script>
Note: The code is heavily commented to explain its workings for educational purposes.

---

## Educational Insights

- **Research and Experimentation:**  
  This project required digging into networking standards and RFC documents—which is an essential part of computer science research. It shows that even simple programs encompass a deep understanding of protocols and data formats.

- **Hands-on Learning:**  
  I gained insights into socket programming, data encoding/decoding, decompression, and error troubleshooting.

- **Real-World Application:**  
  Understanding DNS is crucial for a wide range of IT fields including cybersecurity, network administration, and software development. This project will equip me to better understand Enterprise DNS in the corportate environment.

---

## Conclusion

For my work here, I received extra credit from my professor and went above and beyond the scope of the course.

