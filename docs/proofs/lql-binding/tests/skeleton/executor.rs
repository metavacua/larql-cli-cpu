// Skeleton of larql-lql's executor shape, for testing the binding-facts tooling.
pub enum Statement {
    ShowModels,
    BeginPatch { path: String },
    Stats,
    Walk { prompt: String },
    Hidden,
    Use { path: String },
}

impl Session {
    pub fn execute(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::ShowModels => self.exec_show_models(),
            Statement::BeginPatch { path } => self.exec_begin_patch(path),
            Statement::Stats => self.exec_stats(),
            Statement::Walk { prompt } => self.exec_walk(prompt),
            Statement::Hidden => self.exec_hidden(),
            Statement::Use { path } => self.exec_use(path),
        }
    }

    fn exec_show_models(&self) -> Result<Vec<String>, LqlError> {
        Ok(list_dir())
    }

    fn exec_begin_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        self.recording = Some(path.to_string());
        Ok(vec![])
    }

    fn exec_stats(&self) -> Result<Vec<String>, LqlError> {
        match &self.backend {
            Backend::None => Err(LqlError::NoBackend),
            _ => Ok(vec![]),
        }
    }

    fn exec_walk(&self, _prompt: &str) -> Result<Vec<String>, LqlError> {
        let _v = self.require_vindex()?;
        Ok(vec![])
    }

    fn require_vindex(&self) -> Result<&Vindex, LqlError> {
        match &self.backend {
            Backend::Vindex(v) => Ok(v),
            _ => Err(LqlError::NoBackend),
        }
    }

    // The only call is inside a macro argument: invisible to a syntactic pass.
    fn exec_hidden(&self) -> Result<Vec<String>, LqlError> {
        Ok(vec![format!("{}", self.peek())])
    }

    fn peek(&self) -> String {
        match &self.backend {
            _ => String::new(),
        }
    }

    fn exec_use(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        self.backend = Backend::open(path);
        Ok(vec![])
    }
}

fn list_dir() -> Vec<String> {
    vec![]
}

#[cfg(test)]
mod tests {
    // Same name as a handler, and reads the backend: must not count.
    impl super::Session {
        fn exec_show_models(&self) -> bool {
            let b = &self.backend;
            b.is_none()
        }
    }
}
