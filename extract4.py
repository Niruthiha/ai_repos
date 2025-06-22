#!/usr/bin/env python3
"""
Enhanced LLM Application Repository Finder
Finds actual APPLICATION repositories that USE LLM services (OpenAI, Google AI, Cohere, etc.)
Focuses on real applications rather than frameworks/libraries, even with lower star counts.
"""

import os
import json
import time
import requests
import argparse
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAppFinder:
    def __init__(self, github_token: str, output_dir: str):
        if not github_token:
            raise ValueError("GitHub token cannot be empty.")
        self.token = github_token
        self.output_dir = output_dir
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        os.makedirs(self.output_dir, exist_ok=True)

        # Enhanced search queries targeting specific LLM application types
        self.llm_usage_searches = [
            # Document Interaction & Analysis Applications
            {
                'query': 'document chat pdf openai language:python stars:>3',
                'description': 'PDF document chat applications'
            },
            {
                'query': 'semantic search embeddings language:python stars:>3',
                'description': 'Semantic search applications'
            },
            {
                'query': 'RAG OR "retrieval augmented" streamlit language:python stars:>3',
                'description': 'RAG document analysis apps'
            },
            {
                'query': 'document summarization openai language:python stars:>3',
                'description': 'Document summarization applications'
            },
            {
                'query': 'pdf qa chatbot language:python stars:>3',
                'description': 'PDF Q&A chatbot applications'
            },
            {
                'query': 'document analysis llm language:python stars:>3',
                'description': 'Document analysis LLM apps'
            },
            {
                'query': 'information extraction openai language:python stars:>3',
                'description': 'Information extraction applications'
            },
            
            # Customer Support & Chatbot Applications
            {
                'query': 'customer support chatbot openai language:python stars:>3',
                'description': 'Customer support chatbot apps'
            },
            {
                'query': 'helpdesk bot llm language:python stars:>3',
                'description': 'Helpdesk bot applications'
            },
            {
                'query': 'support ticket automation openai language:python stars:>3',
                'description': 'Support ticket automation apps'
            },
            {
                'query': 'discord bot openai language:python stars:>3',
                'description': 'Discord bot applications using LLMs'
            },
            {
                'query': 'telegram bot gpt language:python stars:>3',
                'description': 'Telegram bot applications'
            },
            {
                'query': 'slack bot openai language:python stars:>3',
                'description': 'Slack bot applications'
            },
            
            # Code Generation & Development Tools
            {
                'query': 'code generator openai language:python stars:>3',
                'description': 'Code generation applications'
            },
            {
                'query': 'code assistant llm language:python stars:>3',
                'description': 'Code assistant applications'
            },
            {
                'query': 'code review bot openai language:python stars:>3',
                'description': 'Code review automation tools'
            },
            {
                'query': 'unit test generator gpt language:python stars:>3',
                'description': 'Unit test generation tools'
            },
            {
                'query': 'code explanation llm language:python stars:>3',
                'description': 'Code explanation applications'
            },
            
            # Web Applications with Common Frameworks
            {
                'query': 'OPENAI_API_KEY streamlit language:python stars:>3',
                'description': 'Streamlit apps with OpenAI integration'
            },
            {
                'query': 'OPENAI_API_KEY gradio language:python stars:>3',
                'description': 'Gradio apps with OpenAI integration'
            },
            {
                'query': 'openai flask language:python stars:>3',
                'description': 'Flask web apps using OpenAI'
            },
            {
                'query': 'openai fastapi language:python stars:>3',
                'description': 'FastAPI apps using OpenAI'
            },
            {
                'query': 'openai django language:python stars:>3',
                'description': 'Django apps using OpenAI'
            },
            
            # AI Agents & Workflow Automation
            {
                'query': 'ai agent openai language:python stars:>3',
                'description': 'AI agent applications'
            },
            {
                'query': 'workflow automation llm language:python stars:>3',
                'description': 'Workflow automation with LLMs'
            },
            {
                'query': 'email automation openai language:python stars:>3',
                'description': 'Email automation applications'
            },
            {
                'query': 'task automation gpt language:python stars:>3',
                'description': 'Task automation applications'
            },
            
            # Google AI / Gemini usage
            {
                'query': 'google.generativeai language:python stars:>3 -google/generative-ai',
                'description': 'Python apps using Google Generative AI'
            },
            {
                'query': 'GOOGLE_AI_KEY OR gemini-pro streamlit language:python stars:>3',
                'description': 'Apps using Google AI/Gemini with Streamlit'
            },
            {
                'query': 'gemini chatbot language:python stars:>3',
                'description': 'Chatbot apps using Gemini'
            },
            
            # Cohere usage
            {
                'query': 'cohere.Client language:python stars:>3 -cohere/cohere-python',
                'description': 'Python apps using Cohere API'
            },
            {
                'query': 'COHERE_API_KEY language:python stars:>3',
                'description': 'Apps with Cohere API usage'
            },
            
            # Anthropic Claude usage
            {
                'query': 'anthropic.Client language:python stars:>3 -anthropic/anthropic-sdk',
                'description': 'Apps using Anthropic Claude API'
            },
            {
                'query': 'ANTHROPIC_API_KEY language:python stars:>3',
                'description': 'Apps with Claude API usage'
            },
            {
                'query': 'claude chatbot language:python stars:>3',
                'description': 'Chatbot apps using Claude'
            },
            
            # Other LLM providers
            {
                'query': 'together.ai OR together_ai language:python stars:>3',
                'description': 'Apps using Together AI'
            },
            {
                'query': 'replicate.run language:python stars:>3 -replicate/replicate-python',
                'description': 'Apps using Replicate API'
            },
            {
                'query': 'REPLICATE_API_TOKEN language:python stars:>3',
                'description': 'Apps with Replicate API usage'
            },
            {
                'query': 'huggingface_hub.InferenceClient language:python stars:>3',
                'description': 'Apps using HuggingFace Inference API'
            },
            
            # Application-specific patterns
            {
                'query': 'RAG OR "retrieval augmented" streamlit language:python stars:>5',
                'description': 'RAG applications with Streamlit'
            },
            {
                'query': 'document chat openai language:python stars:>3',
                'description': 'Document chat applications'
            },
            {
                'query': 'pdf chat gpt language:python stars:>3',
                'description': 'PDF chat applications'
            },
            {
                'query': 'voice assistant openai language:python stars:>3',
                'description': 'Voice assistant applications'
            },
            {
                'query': 'text summarization openai language:python stars:>3',
                'description': 'Text summarization apps'
            },
            {
                'query': 'code generator openai language:python stars:>3',
                'description': 'Code generation applications'
            },
            
            # JavaScript/TypeScript LLM apps (lower thresholds)
            {
                'query': 'openai nextjs javascript stars:>5',
                'description': 'Next.js apps using OpenAI'
            },
            {
                'query': 'OPENAI_API_KEY react javascript stars:>5',
                'description': 'React apps with OpenAI API'
            },
            {
                'query': 'openai express nodejs stars:>3',
                'description': 'Express.js apps using OpenAI'
            },
            {
                'query': 'chatbot openai typescript stars:>3',
                'description': 'TypeScript chatbot apps'
            },
            
            # Web applications patterns
            {
                'query': 'openai web app python stars:>3',
                'description': 'Python web applications using OpenAI'
            },
            {
                'query': 'llm dashboard streamlit stars:>3',
                'description': 'LLM dashboard applications'
            },
            {
                'query': 'ai assistant flask python stars:>3',
                'description': 'AI assistant Flask applications'
            }
        ]

    def _make_request(self, url: str, params: dict = None) -> dict or None:
        """Request with enhanced error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 10:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_duration = max(reset_time - time.time(), 60)
                        logger.warning(f"Rate limit critical. Sleeping {sleep_duration:.0f}s")
                        time.sleep(sleep_duration)

                response.raise_for_status()
                return response.json()

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(15)
                else:
                    return None
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        return None

    def search_llm_applications(self) -> list:
        """Search for repositories that actually USE LLM services."""
        logger.info("🔍 Searching for applications that USE LLM services...")
        
        found_repos = set()
        all_search_results = []
        
        for search_config in self.llm_usage_searches:
            query = search_config['query']
            description = search_config['description']
            
            logger.info(f"🔎 Searching: {description}")
            logger.info(f"   Query: {query}")
            
            # Search repositories
            search_url = f"{self.base_url}/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 30  # Increased to get more results
            }
            
            search_results = self._make_request(search_url, params)
            
            if not search_results or 'items' not in search_results:
                logger.warning(f"   ❌ No results found")
                time.sleep(2)
                continue
            
            repos_found = 0
            for repo in search_results['items']:
                repo_name = repo['full_name']
                
                # Skip if already found
                if repo_name in found_repos:
                    continue
                
                # Enhanced filtering for actual applications
                if self._is_framework_or_library(repo):
                    continue
                
                found_repos.add(repo_name)
                all_search_results.append({
                    'repo': repo_name,
                    'stars': repo['stargazers_count'],
                    'language': repo.get('language', ''),
                    'description': (repo.get('description') or '')[:150],
                    'search_query': query,
                    'search_description': description,
                    'url': repo['html_url'],
                    'topics': repo.get('topics', []),
                    'size': repo.get('size', 0),
                    'default_branch': repo.get('default_branch', 'main')
                })
                repos_found += 1
            
            logger.info(f"   ✅ Found {repos_found} new application repos")
            time.sleep(3)  # Be respectful to search API
        
        # Sort by stars and remove duplicates
        unique_repos = sorted(all_search_results, key=lambda x: x['stars'], reverse=True)
        
        logger.info(f"\n📊 SEARCH SUMMARY:")
        logger.info(f"   • Total unique repositories found: {len(unique_repos)}")
        logger.info(f"   • Search queries executed: {len(self.llm_usage_searches)}")
        
        return unique_repos

    def _is_framework_or_library(self, repo: dict) -> bool:
        """Enhanced filtering to identify frameworks/libraries vs applications."""
        repo_name = repo['full_name'].lower()
        description = (repo.get('description') or '').lower()
        topics = [topic.lower() for topic in repo.get('topics', [])]
        
        # Official/organizational repositories (likely to be frameworks)
        official_orgs = [
            'openai/', 'anthropic/', 'google/', 'microsoft/', 'cohere-ai/',
            'langchain-ai/', 'huggingface/', 'gradio-app/', 'streamlit/',
            'replicate/', 'togetherai/', 'pinecone-io/'
        ]
        
        for org in official_orgs:
            if repo_name.startswith(org):
                return True
        
        # Framework/library naming patterns
        framework_patterns = [
            # Direct framework indicators
            'python-sdk', 'sdk', 'api-client', 'client', 'wrapper', 'library',
            'framework', 'toolkit', 'package', 'binding', 'adapter',
            
            # Specific framework names
            'langchain', 'semantic-kernel', 'llama-index', 'transformers',
            'sentence-transformers', 'diffusers', 'accelerate', 'tokenizers',
            
            # Generic tools (not specific applications)
            '/utils', '/tools', '/helpers', '/common', '/core', '/base'
        ]
        
        for pattern in framework_patterns:
            if pattern in repo_name or pattern in description:
                return True
        
        # Topic-based filtering
        framework_topics = [
            'sdk', 'api', 'library', 'framework', 'package', 'toolkit',
            'python-package', 'npm-package', 'pip', 'pypi'
        ]
        
        for topic in framework_topics:
            if topic in topics:
                return True
        
        # Description-based filtering for frameworks
        framework_descriptions = [
            'python client', 'api wrapper', 'sdk for', 'library for',
            'framework for', 'toolkit for', 'package for', 'binding for',
            'official client', 'client library', 'python package'
        ]
        
        for desc_pattern in framework_descriptions:
            if desc_pattern in description:
                return True
        
        # Additional check: if repo name contains common app indicators, it's likely an app
        app_indicators = [
            'chatbot', 'chat-', 'assistant', 'app', 'web', 'dashboard',
            'ui', 'frontend', 'backend', 'server', 'client-app', 'demo',
            'example', 'tutorial', 'project', 'poc', 'prototype'
        ]
        
        has_app_indicator = any(indicator in repo_name for indicator in app_indicators)
        
        # If it has app indicators, it's likely an application
        if has_app_indicator:
            return False
        
        return False

    def analyze_repo_content(self, repo_name: str) -> dict:
        """Analyze repository content to confirm it's an actual LLM application."""
        # Get repository contents
        contents_url = f"{self.base_url}/repos/{repo_name}/contents"
        contents = self._make_request(contents_url)
        
        if not contents:
            return {"is_app": False, "reason": "CANNOT_FETCH_CONTENTS"}
        
        # Look for application indicators in file structure
        files = [item['name'].lower() for item in contents if item['type'] == 'file']
        dirs = [item['name'].lower() for item in contents if item['type'] == 'dir']
        
        # Application file indicators (more comprehensive)
        app_files = [
            # Python web applications
            'app.py', 'main.py', 'server.py', 'run.py', 'start.py', 'wsgi.py',
            'streamlit_app.py', 'gradio_app.py', 'flask_app.py', 'fastapi_app.py',
            
            # Configuration and deployment
            'requirements.txt', 'pyproject.toml', 'environment.yml', 'conda.yml',
            'dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
            'config.py', 'settings.py', '.env.example', 'env.example',
            
            # Web application files
            'package.json', 'index.js', 'server.js', 'app.js',
            
            # Documentation and README
            'readme.md', 'readme.rst', 'readme.txt',
            
            # Common application patterns
            'bot.py', 'chatbot.py', 'assistant.py', 'agent.py',
            'chat.py', 'api.py', 'routes.py', 'views.py'
        ]
        
        # Web application directories (expanded)
        app_dirs = [
            # Frontend directories
            'templates', 'static', 'public', 'assets', 'css', 'js', 'images',
            'src', 'components', 'pages', 'views', 'layouts',
            
            # Backend directories
            'routes', 'api', 'controllers', 'models', 'services', 'utils',
            'handlers', 'middleware', 'auth', 'core',
            
            # Application-specific directories
            'agents', 'bots', 'chat', 'docs', 'data', 'uploads',
            'config', 'tests', 'scripts'
        ]
        
        # Framework/library indicators (negative signals)
        framework_files = [
            'setup.py', 'pyproject.toml', '__init__.py', 'setup.cfg',
            'manifest.in', 'tox.ini', 'pytest.ini'
        ]
        
        app_file_score = sum(1 for f in app_files if f in files)
        app_dir_score = sum(1 for d in app_dirs if d in dirs)
        framework_score = sum(1 for f in framework_files if f in files)
        
        # Check for specific LLM usage in main files
        llm_usage_score = 0
        main_files = ['app.py', 'main.py', 'server.py', 'requirements.txt']
        
        for filename in main_files:
            if filename in files:
                try:
                    file_url = f"{self.base_url}/repos/{repo_name}/contents/{filename}"
                    file_data = self._make_request(file_url)
                    if file_data and 'content' in file_data:
                        import base64
                        content = base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')
                        
                        # Check for LLM API usage
                        llm_patterns = [
                            'openai', 'anthropic', 'cohere', 'together', 'replicate',
                            'google.generativeai', 'gemini', 'claude', 'gpt-3.5', 'gpt-4',
                            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'
                        ]
                        
                        llm_usage_score += sum(1 for pattern in llm_patterns if pattern.lower() in content.lower())
                except:
                    pass
        
        # Decision logic
        is_app = (
            (app_file_score >= 2 or app_dir_score >= 2) and
            llm_usage_score >= 1 and
            framework_score < 3
        )
        
        return {
            "is_app": is_app,
            "scores": {
                "app_files": app_file_score,
                "app_dirs": app_dir_score,
                "framework_files": framework_score,
                "llm_usage": llm_usage_score
            },
            "files": files[:10],  # First 10 files for debugging
            "dirs": dirs[:10]     # First 10 directories for debugging
        }

    def analyze_repo_llm_usage(self, repo_name: str) -> dict:
        """Analyze a repository for BOTH LLM usage AND good label practices."""
        # Get repository information
        repo_data = self._make_request(f"{self.base_url}/repos/{repo_name}")
        if not repo_data:
            return {"suitable": False, "reason": "API_ERROR"}
        
        # Enhanced content analysis
        content_analysis = self.analyze_repo_content(repo_name)
        if not content_analysis["is_app"]:
            return {
                "suitable": False, 
                "reason": "NOT_AN_APPLICATION",
                "content_analysis": content_analysis
            }
        
        # Check if issues are enabled
        if not repo_data.get('has_issues', False):
            return {"suitable": False, "reason": "NO_ISSUES_ENABLED"}
        
        issue_count = repo_data.get('open_issues_count', 0)
        # Lowered threshold for small applications
        if issue_count < 2:
            return {"suitable": False, "reason": "TOO_FEW_ISSUES", "issue_count": issue_count}
        
        # Sample recent issues to check for BOTH LLM usage AND label practices
        issues_url = f"{self.base_url}/repos/{repo_name}/issues"
        params = {
            'state': 'all',
            'per_page': 20,
            'sort': 'updated',
            'direction': 'desc'
        }
        
        recent_issues = self._make_request(issues_url, params)
        if not recent_issues:
            return {"suitable": False, "reason": "CANNOT_FETCH_ISSUES"}
        
        # Filter out pull requests
        actual_issues = [issue for issue in recent_issues if 'pull_request' not in issue]
        
        if len(actual_issues) < 2:  # Lowered threshold
            return {"suitable": False, "reason": "TOO_FEW_ACTUAL_ISSUES"}
        
        # Enhanced LLM usage keywords based on application types
        llm_keywords = [
            # API and Technical Issues
            'api key', 'rate limit', 'token', 'authentication', 'quota exceeded',
            'timeout', 'connection error', 'api error', 'invalid key', 'billing',
            
            # LLM Provider Names
            'openai', 'gpt', 'claude', 'gemini', 'cohere', 'together', 'replicate',
            'anthropic', 'google ai', 'generative ai', 'huggingface', 'azure openai',
            
            # LLM Concepts and Operations
            'llm', 'model', 'prompt', 'completion', 'chat', 'generation', 'embedding',
            'fine-tuning', 'temperature', 'max tokens', 'context window', 'system prompt',
            
            # Application-Specific Terms
            'document chat', 'pdf processing', 'semantic search', 'rag', 'retrieval',
            'summarization', 'question answering', 'information extraction',
            'code generation', 'code review', 'unit test', 'debugging',
            'chatbot', 'conversation', 'customer support', 'helpdesk',
            'workflow automation', 'task automation', 'email processing',
            'meeting transcription', 'content generation', 'blog generation',
            
            # Technical Issues Common in LLM Apps
            'hallucination', 'context length', 'token limit', 'memory usage',
            'response time', 'accuracy', 'prompt engineering', 'chain of thought',
            'few-shot', 'zero-shot', 'fine-tune', 'model performance',
            
            # General Error Terms
            'error', 'bug', 'issue', 'problem', 'fix', 'help', 'not working',
            'fails', 'crash', 'exception', 'troubleshoot'
        ]
        
        llm_related_issues = 0
        for issue in actual_issues:
            title = (issue.get('title') or '').lower()
            body = (issue.get('body') or '').lower()
            combined_text = title + ' ' + body
            
            if any(keyword in combined_text for keyword in llm_keywords):
                llm_related_issues += 1
        
        llm_percentage = (llm_related_issues / len(actual_issues)) * 100
        has_llm_usage = llm_related_issues >= 1 or llm_percentage >= 10  # Lowered threshold
        
        # ===== CHECK 2: LABEL USAGE ANALYSIS (More lenient) =====
        # Calculate label statistics
        issues_with_labels = sum(1 for issue in actual_issues if issue.get('labels'))
        total_labels = sum(len(issue.get('labels', [])) for issue in actual_issues)
        label_percentage = (issues_with_labels / len(actual_issues)) * 100
        avg_labels_per_issue = total_labels / len(actual_issues)
        
        # Get repository labels to see if they have an organized label system
        labels_url = f"{self.base_url}/repos/{repo_name}/labels"
        repo_labels = self._make_request(labels_url)
        total_repo_labels = len(repo_labels) if repo_labels else 0
        
        # More lenient labeling requirements for smaller apps
        has_good_labels = (
            label_percentage >= 20 or  # At least 20% of issues have labels
            total_repo_labels >= 3 or  # Repository has at least 3 different labels
            avg_labels_per_issue >= 0.3  # Average 0.3 labels per issue
        )
        
        # ===== FINAL DECISION: PRIORITIZE LLM USAGE =====
        # For research purposes, prioritize LLM usage over perfect labeling
        suitable = has_llm_usage and (has_good_labels or llm_percentage >= 30)
        
        # Determine specific reason
        if not has_llm_usage and not has_good_labels:
            reason = "NO_LLM_USAGE_AND_POOR_LABELS"
        elif not has_llm_usage:
            reason = "NO_LLM_USAGE"
        elif not has_good_labels and llm_percentage < 30:
            reason = "POOR_LABELING_AND_LOW_LLM_USAGE"
        else:
            reason = "GOOD_LLM_APP"
        
        return {
            "suitable": suitable,
            "reason": reason,
            "content_analysis": content_analysis,
            "stats": {
                "total_issues_checked": len(actual_issues),
                "llm_related_issues": llm_related_issues,
                "llm_percentage": round(llm_percentage, 1),
                "issues_with_labels": issues_with_labels,
                "label_percentage": round(label_percentage, 1),
                "avg_labels_per_issue": round(avg_labels_per_issue, 2),
                "total_repo_labels": total_repo_labels,
                "repo_stars": repo_data.get('stargazers_count', 0),
                "repo_language": repo_data.get('language', ''),
                "repo_description": (repo_data.get('description') or '')[:100],
                "created_at": repo_data.get('created_at', ''),
                "updated_at": repo_data.get('updated_at', '')
            }
        }

    def extract_issues_from_llm_apps(self, repo_candidates: list) -> list:
        """Extract issues from LLM application repositories that ALSO have good labeling."""
        logger.info("🏷️  Analyzing LLM applications for AI usage and labeling...")
        
        suitable_repos = []
        detailed_analysis = []
        
        for i, repo_info in enumerate(repo_candidates, 1):
            repo_name = repo_info['repo']
            logger.info(f"🔍 [{i}/{len(repo_candidates)}] Analyzing {repo_name}...")
            
            analysis = self.analyze_repo_llm_usage(repo_name)
            analysis['search_info'] = repo_info
            detailed_analysis.append({"repo": repo_name, **analysis})
            
            if analysis["suitable"]:
                suitable_repos.append(repo_name)
                stats = analysis["stats"]
                content = analysis.get("content_analysis", {})
                logger.info(f"✅ {repo_name}: {stats['llm_related_issues']} LLM issues "
                           f"({stats['llm_percentage']}%), {stats['label_percentage']}% labeled, "
                           f"{stats['repo_stars']} stars, App Score: {content.get('scores', {})}")
            else:
                if "stats" in analysis:
                    stats = analysis["stats"]
                    logger.info(f"❌ {repo_name}: {analysis['reason']} "
                               f"(LLM: {stats['llm_percentage']}%, Labels: {stats['label_percentage']}%)")
                else:
                    logger.info(f"❌ {repo_name}: {analysis['reason']}")
            
            time.sleep(2)  # Be respectful to API
        
        # Save detailed analysis
        analysis_file = os.path.join(self.output_dir, 'llm_app_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(detailed_analysis, f, indent=2)
        
        logger.info(f"\n📊 LLM APPLICATION ANALYSIS:")
        logger.info(f"   • Candidates analyzed: {len(repo_candidates)}")
        logger.info(f"   • Suitable repos (Apps with LLM usage): {len(suitable_repos)}")
        logger.info(f"   • Analysis saved to: {analysis_file}")
        
        return suitable_repos

    def extract_llm_issues(self, repo_name: str, max_issues: int = 200) -> list:
        """Extract issues from LLM applications with focus on API/service issues."""
        logger.info(f"🔧 Extracting LLM-related issues from {repo_name}...")
        
        issues_data = []
        since_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        params = {
            'state': 'all',
            'since': since_date,
            'per_page': 100,
            'direction': 'desc',
            'sort': 'updated'
        }
        
        url = f"{self.base_url}/repos/{repo_name}/issues"
        page = 1
        
        while len(issues_data) < max_issues:
            params['page'] = page
            issues = self._make_request(url, params)
            
            if not issues:
                break
            
            for issue in tqdm(issues, desc=f"Processing {repo_name} issues (Page {page})"):
                if 'pull_request' in issue:
                    continue
                
                labels = [label.get('name', '') for label in issue.get('labels', [])]
                
                # Get limited comments
                comments = []
                if issue.get('comments', 0) > 0:
                    comments_data = self._make_request(issue['comments_url'])
                    if comments_data:
                        comments = [{'user': c['user']['login'], 'body': c['body']} 
                                  for c in comments_data[:3]]
                
                issues_data.append({
                    'issue_number': issue['number'],
                    'title': issue['title'],
                    'author': issue['user']['login'],
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at'],
                    'labels': labels,
                    'body': issue['body'],
                    'comments': comments,
                    'repository': repo_name
                })
                
                if len(issues_data) >= max_issues:
                    break
            
            if len(issues) < 100:
                break
            
            page += 1
            time.sleep(1)
        
        logger.info(f"✅ Extracted {len(issues_data)} issues from {repo_name}")
        return issues_data

    def run_llm_app_extraction(self):
        """Main extraction workflow for LLM application repositories."""
        logger.info("🚀 Starting Enhanced LLM Application Repository Extraction...")
        
        # 1. Search for LLM application repositories
        repo_candidates = self.search_llm_applications()
        
        if not repo_candidates:
            logger.error("❌ No LLM application repositories found!")
            return
        
        # Save search results
        search_results_file = os.path.join(self.output_dir, 'llm_app_search_results.json')
        with open(search_results_file, 'w') as f:
            json.dump(repo_candidates, f, indent=2)
        
        # 2. Analyze and filter repositories
        suitable_repos = self.extract_issues_from_llm_apps(repo_candidates)
        
        if not suitable_repos:
            logger.error("❌ No suitable LLM application repositories found!")
            return
        
        # 3. Extract issues from suitable repositories
        logger.info(f"\n🎯 Extracting issues from {len(suitable_repos)} LLM applications...")
        
        total_issues = 0
        successful_extractions = 0
        
        for i, repo_name in enumerate(suitable_repos, 1):
            logger.info(f"\n📖 Processing {i}/{len(suitable_repos)}: {repo_name}")
            
            # Check if already processed
            safe_filename = repo_name.replace('/', '_') + '_llm_issues.json'
            output_path = os.path.join(self.output_dir, safe_filename)
            
            if os.path.exists(output_path):
                logger.info(f"⏭️  Already processed - skipping")
                continue
            
            # Extract issues
            repo_issues = self.extract_llm_issues(repo_name)
            
            if repo_issues:
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(repo_issues, f, indent=2, ensure_ascii=False)
                
                successful_extractions += 1
                total_issues += len(repo_issues)
                logger.info(f"💾 Saved {len(repo_issues)} issues")
            
            time.sleep(3)
        
        # Final summary
        logger.info(f"\n🎉 ENHANCED LLM APPLICATION EXTRACTION COMPLETED!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   • LLM applications processed: {successful_extractions}")
        logger.info(f"   • Total issues extracted: {total_issues}")
        logger.info(f"   • Avg issues per app: {total_issues/max(successful_extractions,1):.1f}")
        logger.info(f"   • Search results saved to: {search_results_file}")
        logger.info(f"   • ✅ Focus on actual applications using LLM APIs!")

def main():
    parser = argparse.ArgumentParser(description='Enhanced LLM Application Repository Finder')
    parser.add_argument('--token', required=True, help='GitHub Personal Access Token')
    parser.add_argument('--output_dir', default='./llm_app_data', help='Output directory')
    
    args = parser.parse_args()
    
    finder = LLMAppFinder(github_token=args.token, output_dir=args.output_dir)
    finder.run_llm_app_extraction()

if __name__ == "__main__":
    main()